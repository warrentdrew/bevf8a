# !/usr/bin/env python3
"""
Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import pyquaternion
import tempfile
import pickle
from projects.nuscenes.utils.data_classes import Box as NuScenesBox
from projects.nuscenes.eval.detection.config import config_factory
from projects.nuscenes.eval.detection.data_classes import DetectionConfig
from os import path as osp
from projects.nuscenes.eval.detection.evaluate import MyDataEval 

from paddle3d.apis import manager

from .core import load, dump, mkdir_or_exist
from .custom_3d import Custom3DDataset

from paddle3d.geometries import BBoxes2D, BBoxes3D


@manager.DATASETS.add_component
class AT128DatasetValid(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
   
    CLASSES = ('bigMot', 'smallMot', 'nonMot', 'pedestrian', 'TrainedOthers')
    def __init__(self,
                 ann_file,
                 num_views=6,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 test_gt=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False, # True: only consider object that appear in the image and point cloud; False: consider all object
                 # Add
                 extrinsics_noise=False,
                 extrinsics_noise_type='single',
                 drop_frames=False,
                 drop_set=[0,'discrete'],
                 noise_sensor_type='camera',
                 eval_configs = None,
                 cam_orders=['spherical-right-forward', 'spherical-right-backward', 'spherical-left-forward', 
                    'spherical-left-backward', 'spherical-backward', 'obstacle']):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.cam_orders = cam_orders
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.num_views = num_views
        assert self.num_views <= 6
        self.with_velocity = with_velocity
        self.eval_version = eval_version
       
        metric = eval_configs.pop('metric')
        self.render_curves = eval_configs.pop('render_curves')

        if 'lidar' in metric:
            self.lidar_metric = True
        else:
            self.lidar_metric = False
        if 'camera' in metric:
            self.camera_metric = True
        else:
            self.camera_metric = False

        self.eval_detection_configs = DetectionConfig.deserialize(eval_configs)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        ### for frop foreground points
        self.test_gt = test_gt
        self.use_valid_flag = use_valid_flag
        ## 增加部分
        self.extrinsics_noise = extrinsics_noise # 外参是否扰动
        assert extrinsics_noise_type in ['all', 'single'] 
        self.extrinsics_noise_type = extrinsics_noise_type # 外参扰动类型
        self.drop_frames = drop_frames # 是否丢帧
        self.drop_ratio = drop_set[0] # 丢帧比例：assert ratio in [10, 20, ..., 90]
        self.drop_type = drop_set[1] # 丢帧情况：连续(consecutive) or 离散(discrete)
        self.noise_sensor_type = noise_sensor_type # lidar or camera 丢帧

        if self.extrinsics_noise or self.drop_frames:
            pkl_file = open('./data/nuscenes/nuscenes_infos_val_with_noise.pkl', 'rb')
            noise_data = pickle.load(pkl_file)
            self.noise_data = noise_data[noise_sensor_type]
        else:
            self.noise_data = None
        
        print('noise setting:')
        #print("self.CLASSES: ", self.CLASSES)
        if self.drop_frames:
            print('frame drop setting: drop ratio:', self.drop_ratio, ', sensor type:', self.noise_sensor_type, ', drop type:', self.drop_type)
        if self.extrinsics_noise:
            assert noise_sensor_type=='camera'
            print(f'add {extrinsics_noise_type} noise to extrinsics')
    
    ### for frop foreground points
    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode or self.test_gt:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        # if self.use_valid_flag:
        #     mask = info['valid_flag']
        #     gt_names = set(info['gt_names'][mask])
        # else:
        #     gt_names = set(info['gt_names'])
        gt_names = set(info['gt_names'])
        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        print("ann_file: ", ann_file)
        data_infos = pickle.load(open(ann_file, 'rb'), encoding='iso-8859-1')
        data_infos = list(sorted(data_infos['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index] # dict_keys(['gt_boxes', 'gt_names', 'timestamp', 'cams', 'valid_flag'])
        
        input_dict = dict(
            timestamp=info['timestamp'] / 1e6,
        )

        #cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            caminfos = []
            cam_intrinsics = []
            for cam_type in self.cam_orders:
                cam_info = info['cams'][cam_type]
                if 'data_path' not in cam_info:
                    return None
                each_path = cam_info['data_path']
                image_paths.append(each_path)

                lidar2cam = cam_info['lidar2cam'] 
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:4, :4] = lidar2cam

                intrinsic = np.array(cam_info['cam_intrinsic'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam) 
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt)

                caminfos.append(
                    {
                    'cam_intrinsic':cam_info['cam_intrinsic']
                    })
            if len(image_paths) != len(self.cam_orders): 
                print("image_paths: ", image_paths)
                print("self.cam_orders: ", self.cam_orders)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    caminfo=caminfos,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsic=cam_intrinsics,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict


    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d'] != -1).any()):
            return None
        if self.use_valid_flag:
            if sum(example['valid_flag']) == 0:
                return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        valid_flag = np.array([i for i in info['valid_flag']]) 
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = np.zeros((gt_bboxes_3d.shape[0], 2))
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d[:, :3] += gt_bboxes_3d[:, 3:6] * \
                (np.array([0.5, 0.5, 0.0]) - np.array([0.5, 0.5, 0.5]))
        gt_bboxes_3d = BBoxes3D(
            gt_bboxes_3d,
            origin=(0.5, 0.5, 0.5))

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            valid_flag = valid_flag)
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        print("mapped_class_names: ", mapped_class_names)
        for sample_id, det in enumerate(results):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = str(self.data_infos[sample_id]['timestamp'])

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                attr = 'vehicle.moving'

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_at128.json')
        print('Results writes to', res_path)
        dump(nusc_submissions, res_path)
        dump(nusc_submissions, './results_at128.json')

        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
      

        output_dir = osp.join(*osp.split(result_path)[:-1])
        print("result_path (_evaluate_single): ",result_path)
        nusc_eval = MyDataEval(
            None,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=self.ann_file,
            output_dir=output_dir,
            verbose=True)
        nusc_eval.main(render_curves=self.render_curves,
                       lidar_metric=self.lidar_metric,
                       camera_metric=self.camera_metric)


        # record metrics
        #metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        #metric_prefix = f'{result_name}_NuScenes'
        #for name in self.CLASSES:
        #    for k, v in metrics['label_aps'][name].items():
        #        val = float('{:.4f}'.format(v))
        #        detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
        #    for k, v in metrics['label_tp_errors'][name].items():
        #        val = float('{:.4f}'.format(v))
        #        detail['{}/{}_{}'.format(metric_prefix, name, k)] = val

        #detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        #detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        
        if not isinstance(results[0], dict):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 #jsonfile_prefix='./work_dirs/',
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        print(results_dict)
        if osp.exists('/evaluation_result/'):
            with open("/evaluation_result/total", "a") as result_file:
                result_file.write("{}\n".format("\n".join(["{}:{}".format(k, v) for k, v in results_dict.items()])))
        return results_dict

def get_gravity_center(bboxes):
    bottom_center = bboxes[:, :3]
    gravity_center = np.zeros_like(bottom_center)
    gravity_center[:, :2] = bottom_center[:, :2]
    gravity_center[:, 2] = bottom_center[:, 2] + bboxes[:, 5] * 0.5
    return gravity_center

def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (paddle.Tensor): Detection scores.
            - labels_3d (paddle.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d'].numpy()
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = get_gravity_center(box3d)
    box_dims = box3d[:, 3:6]
    box_yaw = box3d[:, 6]
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        # TODO: how to select the score threshold
        #if scores[i] < 0.233:
        #    continue
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (0.0, 0.0)

        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list

