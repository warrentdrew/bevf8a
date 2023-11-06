import copy
import numpy as np
import os

from paddle3d.apis import manager
from paddle3d.datasets.at128.core import load
from paddle3d.utils.logger import logger
from paddle3d.utils_idg.box_np_ops import center_to_corner_box2d, box_collision_test


class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True, # TODO yipin close for match result
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]

@manager.TRANSFORMS.add_component
class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 points_loader=None):
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = points_loader

        db_infos = load(info_path)

        # filter database infos
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True) #TODO yipin #shuffle=True)
        # TODO: No group_sampling currently
        print("dbsample.py, self.sample_classes: ", self.sample_classes)
        print("dbsample.py, self.cat2label: ", self.cat2label)

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info.get('sampled_points_num', info['num_points_in_gt']) >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def translate(self, tensor, trans_vector):
        # trans_vector = trans_vector.squeeze(0)
        if len(trans_vector.shape) == 1:
            assert trans_vector.shape[0] == 3
        elif len(trans_vector.shape) == 2:
            assert trans_vector.shape[0] == tensor.shape[0] and \
                trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(
                'Unsupported translation vector of shape {}'.format(
                    trans_vector.shape))
        tensor[:, :3] += trans_vector
        return tensor

    def sample_all(self, gt_bboxes, gt_names, img=None, ground_plane=None, noise_classes=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.
            img (np.ndarray, optional): Image array. Defaults to None.
            ground_plane (np.ndarray, optional): Ground plane information.
                Defaults to None.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels
                  of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
                  sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        if noise_classes is None:
            noise_classes = set()
        # for class_name, max_sample_num in zip(self.sample_classes,
        #                                       self.sample_max_nums):
        # 8A
        for each_rate, class_name, max_sample_num in zip(self.rate , self.sample_classes,
                                        self.sample_max_nums):
            # class_label = self.cat2label[class_name]
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(each_rate * sampled_num).astype(np.int64) # each rate
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    sampled_gt_box = np.concatenate(
                        [s["box3d_lidar"].reshape((-1, 7)) for s in sampled_cls], axis=0
                    )
                    collision_box = np.concatenate(
                        [s.get('collision_box', s["box3d_lidar"]).reshape((-1, 7)) for s in sampled_cls], axis=0
                    )

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, collision_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            num_sampled = len(sampled_gt_bboxes)
            s_points_list = []
            num_points_list = []
            gt_names = []
            difficulty = []
            sampled_noise_mask = []
            
            count = 0
            for info in sampled:
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(pts_filename=file_path)
                s_points = self.points_loader(results)['points']
                center = info.get('center', info['box3d_lidar'][:3])  # weakly labeled area
                s_points = self.translate(s_points, center)

                count += 1

                s_points_list.append(s_points)
                
                if isinstance(info['name'], np.ndarray):  # weakly labeled area
                    gt_names.extend(info['name'])
                    # difficulty.extend(info['difficulty'])
                    # num_points_list.extend(info['num_points_in_gt'])
                    sampled_noise_mask.append(np.zeros(len(s_points), dtype=np.bool))
                else:
                    gt_names.append(info['name'])
                    # difficulty.append(info['difficulty'])
                    # num_points_list.append(s_points.shape[0])
                    if info['name'] in noise_classes:
                        sampled_noise_mask.append(np.ones(len(s_points), dtype=np.bool))
                    else:
                        sampled_noise_mask.append(np.zeros(len(s_points), dtype=np.bool))

            gt_labels = np.array([self.cat2label[cat] if cat in self.classes else -1 for cat in gt_names],
                                 dtype=np.long)

            if ground_plane is not None:
                xyz = sampled_gt_bboxes[:, :3]
                dz = (ground_plane[:3][None, :] *
                      xyz).sum(-1) + ground_plane[3]
                sampled_gt_bboxes[:, 2] -= dz
                for i, s_points in enumerate(s_points_list):
                    s_points[:, 2].sub_(dz[i])

            ret = {
                'gt_names_3d': np.array(gt_names),  # 8A
                # 'difficulty': np.array(difficulty),
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                np.concatenate(s_points_list),
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled)),
                # 'point_num': np.array([num for num in num_points_list]),
                'collision_boxes': avoid_coll_boxes[gt_bboxes.shape[0]:],
                'sampled_noise_mask': np.concatenate(sampled_noise_mask, 0),
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        # sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        sp_boxes = np.concatenate(
            [s.get('collision_box', s["box3d_lidar"]).reshape((-1, 7)) for s in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
