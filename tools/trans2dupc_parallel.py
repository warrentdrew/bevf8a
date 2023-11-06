# !/usr/bin/env python3
# Copyright 2023 Baidu Inc. All Rights Reserved.
# @author: Guojun Wang (wangguojun01@baidu.com)
# @file: trans2dupc_parallel.py
# @brief: trans2dupc_parallel

'''
transform pkl result file to baidu pointcloud result type
used for dupc benchmark
'''
# !/usr/bin/env python3
import pickle
import time
from pathlib import Path
from pypcd import pypcd
# from mmdet3d.core.bbox import box_np_ops
# from mmcv import Config
# import paddle
from paddle3d.utils_idg import box_np_ops
import argparse
import json

import os
import numpy as np
import zipfile
import multiprocessing


def find_points_path(zip_file_list):
    for each_file in zip_file_list:
        if each_file.endswith('.pcd'):
            return each_file


def load_point(zip_path, rotate45degree):
    with zipfile.ZipFile(zip_path, 'r') as zip_obj:
        zip_file_list = zip_obj.namelist()
        points_path = find_points_path(zip_file_list)
        if points_path in zip_file_list:
            points_obj = zip_obj.open(points_path, 'r')
            points_pcd = pypcd.PointCloud.from_fileobj(points_obj)

    x = points_pcd.pc_data["x"]
    y = points_pcd.pc_data["y"]
    z = points_pcd.pc_data["z"]
    intensity = points_pcd.pc_data["intensity"]
    points = np.c_[x, y, z, intensity]
    nan_mask = np.isnan(points).any(axis=1)
    points = points[~nan_mask]
    points.astype(np.float32)

    if rotate45degree:
        points[:, :3] = box_np_ops.rotation_points_single_angle(
            points[:, :3],
            np.pi / 4,
            axis=2
        )
    return points


def rotation_data(ang, points, boxes):
    """ ang: radian """
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3],
        ang,
        axis=2
    )
    boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        boxes[:, :3],
        ang,
        axis=2
    )
    boxes[:, -1] += (ang)
    return points, boxes


def write_result(points, det_boxes, det_labels, det_scores, filename, rotate45degree, save_path, reserve_ori_wlh):
    det_boxes_ori = det_boxes.copy()
    if rotate45degree:
        det_boxes = rotation_data(-np.pi / 4.0,
                                  points.copy(), det_boxes.copy())

    det_boxes[:, 2] = det_boxes[:, 2] - 1.0
    det_boxes[:, 5] = det_boxes[:, 5] + 3.0
    det_boxes[:, 3] = np.ceil(det_boxes[:, 3] / 0.2) * 0.2
    det_boxes[:, 4] = np.ceil(det_boxes[:, 4] / 0.2) * 0.2

    indices = box_np_ops.points_in_rbbox(points, det_boxes, origin=(0.5, 0.5, 0.5))
    mask = indices.sum(0) > 2

    det_boxes = det_boxes[mask]
    det_labels = det_labels[mask]
    det_scores = det_scores[mask]
    det_boxes_ori = det_boxes_ori[mask]
    indices = indices[:, mask]

    # print("processing %s: %d" % (filename, len(det_boxes)))
    filename = save_path / filename
    fn = open(filename, 'w')
    line = "%d %d\n" % (0, len(det_boxes))
    fn.write(line)
    for i in range(len(det_boxes)):

        sensor_type = "velodyne_64"
        object_id = 0
        track_id = 0
        is_background = 0
        confidence = det_scores[i]
        class_name = det_labels[i]
        bbox_center_x = det_boxes[i][0]
        bbox_center_y = det_boxes[i][1]
        bbox_center_z = det_boxes[i][2] - \
            det_boxes[i][5] / 2.0  # get bottom for eval
        bbox_length = det_boxes[i][4]
        bbox_width = det_boxes[i][3]
        bbox_height = det_boxes[i][5]
        yaw = np.pi / 2 - det_boxes[i][6]
        roll = 0
        pitch = 0
        truncated = 0
        occluded = 0
        velocity_x = 0
        velocity_y = 0
        velocity_z = 0
        points_num = 0  # indice.sum(0)#0

        if reserve_ori_wlh:
            bbox_length_ori = det_boxes_ori[i][4]
            bbox_width_ori = det_boxes_ori[i][3]
            bbox_height_ori = det_boxes_ori[i][5]

            line = "%s %d %d %d %.6f %s %.6f %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %d " % \
                (sensor_type, object_id, track_id, is_background, confidence, class_name, bbox_center_x,
                 bbox_center_y, bbox_center_z, bbox_length, bbox_width, bbox_height, bbox_length_ori, bbox_width_ori, bbox_height_ori, yaw, roll, pitch, truncated,
                 occluded, velocity_x, velocity_y, velocity_z, points_num)
        else:
            line = "%s %d %d %d %.6f %s %.6f %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %d " % \
                (sensor_type, object_id, track_id, is_background, confidence, class_name, bbox_center_x,
                 bbox_center_y, bbox_center_z, bbox_length, bbox_width, bbox_height, yaw, roll, pitch, truncated,
                 occluded, velocity_x, velocity_y, velocity_z, points_num)
        # for point in points_in_box:
        #     line = line + ("%.6g %.6g %.6g %u " % (point[0], point[1], point[2], 0))
        line = line + "0\n"
        fn.write(line)

    fn.close()


def DumpResult(result, conf_thresh, rotate45degree, save_path, reserve_ori_wlh, id2class, trans_headstock, headstock_delta_list):

    zip_path = result['filename'].split('+')[0]
    points = load_point(zip_path, rotate45degree)
    zip_path_splits = zip_path.split('/')
    filename = zip_path_splits[-3] + '_' + zip_path_splits[-1][:-4] + '.pcd'

    # print("test1==1!")
    det_boxes = result["pts_bbox"]['boxes_3d'] # .numpy()  #.tensor.numpy()  # 底面中心
    det_labels = result["pts_bbox"]['labels_3d'] # .numpy() # .cpu().numpy()
    det_scores = result["pts_bbox"]['scores_3d'] # .numpy() #.cpu().numpy()
    det_labels = det_labels.tolist()

    det_boxes[:, 2] = det_boxes[:, 2] + det_boxes[:, 5] * 0.5
    for l in range(len(det_labels)):
        det_labels[l] = id2class[det_labels[l]]
    det_labels = np.array(det_labels, dtype=str)

    if det_boxes.shape[0] == 0:
        print(filename)
        non_head_boxes = det_boxes
        non_head_labels = det_labels
        non_head_scores = det_scores
        write_result(points, non_head_boxes, non_head_labels, non_head_scores,
                     filename, rotate45degree, save_path, reserve_ori_wlh)
        if trans_headstock:
            selected_head_boxes_ = np.zeros([0, 7], dtype=det_boxes.dtype)
            selected_head_labels_ = np.zeros([0], dtype=det_labels.dtype)
            selected_head_scores_ = np.zeros([0], dtype=det_scores.dtype)
            for headstock_delta in headstock_delta_list:
                headstock_save_path = Path(str(save_path) + '_headstock' + str(int(headstock_delta)))
                headstock_save_path.mkdir(exist_ok=True, parents=True)
                write_result(points, selected_head_boxes_, selected_head_labels_, selected_head_scores_,
                         filename, rotate45degree, headstock_save_path, reserve_ori_wlh)
    else:
        non_head_inds = det_labels != 'accessory_main'
        non_head_boxes = det_boxes[non_head_inds]
        non_head_labels = det_labels[non_head_inds]
        non_head_scores = det_scores[non_head_inds]
       
        write_result(points, non_head_boxes, non_head_labels, non_head_scores,
                     filename, rotate45degree, save_path, reserve_ori_wlh)

        if trans_headstock:
            head_inds = ~non_head_inds

            head_boxes = det_boxes[head_inds]
            head_labels = det_labels[head_inds]
            head_scores = det_scores[head_inds]

            if 'combo_uid' in result["pts_bbox"].keys() and result["pts_bbox"]["combo_uid"].shape[0] > 0:
                combo_uid = result["pts_bbox"]["combo_uid"] # .numpy() #.cpu().numpy()
                select_head_inds = combo_uid[combo_uid > -1].astype(np.int)
                selected_head_boxes = head_boxes[select_head_inds].reshape(-1, 7)
                selected_head_labels = head_labels[select_head_inds]
                selected_head_scores = head_scores[select_head_inds]

                bigmot_inds = det_labels == 'bigMot'
                bigmot_boxes = det_boxes[bigmot_inds]

                combo_yaw1 = bigmot_boxes[combo_uid > -1][:, 6]
                combo_yaw2 = selected_head_boxes[:, 6]
                for headstock_delta in headstock_delta_list:
                    headstock_save_path = Path(str(save_path) + '_headstock' + str(int(headstock_delta)))
                    headstock_save_path.mkdir(exist_ok=True, parents=True)
                    select_head_inds_ = abs(
                        np.sin(abs(combo_yaw1 - combo_yaw2))) >= np.sin(headstock_delta/180 * np.pi)
                    selected_head_boxes_ = selected_head_boxes[select_head_inds_]
                    selected_head_labels_ = selected_head_labels[select_head_inds_]
                    selected_head_scores_ = selected_head_scores[select_head_inds_]
                    write_result(points, selected_head_boxes_, selected_head_labels_, selected_head_scores_,
                                filename, rotate45degree, headstock_save_path, reserve_ori_wlh)

            else:
                selected_head_boxes_ = np.zeros([0, 7], dtype=det_boxes.dtype)
                selected_head_labels_ = np.zeros([0], dtype=det_labels.dtype)
                selected_head_scores_ = np.zeros([0], dtype=det_scores.dtype)
                for headstock_delta in headstock_delta_list:
                    headstock_save_path = Path(str(save_path) + '_headstock' + str(int(headstock_delta)))
                    headstock_save_path.mkdir(exist_ok=True, parents=True)
                    write_result(points, selected_head_boxes_, selected_head_labels_, selected_head_scores_,
                                filename, rotate45degree, headstock_save_path, reserve_ori_wlh)


class DumpProcess(multiprocessing.Process):
    def __init__(self, processID, process_num, results, conf_thresh, rotate45degree, save_path, reserve_ori_wlh,
                 id2class, trans_headstock,  headstock_delta_list):
        super().__init__()
        self.processID = processID
        self.results = results
        self.conf_thresh = conf_thresh
        self.rotate45degree = rotate45degree
        self.save_path = save_path
        self.reserve_ori_wlh = reserve_ori_wlh
        self.process_num = process_num
        self.id2class = id2class
        self.trans_headstock = trans_headstock
        self.headstock_delta_list = headstock_delta_list

    def run(self):
        # print("=====test1 ===========")
        count = len(os.listdir(self.save_path))
        for idx, result in enumerate(self.results):
            # print("=====test2 ===========")
            if (idx % self.process_num) == self.processID:
                DumpResult(result, self.conf_thresh, self.rotate45degree,
                           self.save_path, self.reserve_ori_wlh, self.id2class,
                           self.trans_headstock,  self.headstock_delta_list)
            # print("=====test3 ===========")
            if self.processID == 0:
                new_count = len(os.listdir(self.save_path))
                if new_count != count:
                    count = new_count
                    print(new_count, 'frames of data have been converted')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', required=True, help='config path')
    parser.add_argument('--result_path', required=True,
                        help='detection image lists to be tested')
    parser.add_argument(
        '--save_path', default='./save_dupc_results', help='save_path')
    parser.add_argument(
        '--class_names', default="unknow, smallMot, bigMot, nonMot, pedestrian, accessory_main", help='cls1,cls2,cls3')
    parser.add_argument('--conf_thresh', default=0.0, type=float, help='')
    parser.add_argument('--reserve_ori_wlh',
                        action='store_true', default=False, help='')
    parser.add_argument('--process_num', type=int, default=10, help='')
    parser.add_argument('--trans_headstock', action='store_true', help='')
    parser.add_argument('--headstock_delta', default=[0.0, 7.0],
                        type=float, nargs='+', help='headstock angle threshold')

    parser.add_argument(
        '--val_path',
        default="",
        help='validation data path'
    )
    args = parser.parse_args()
    print('parsed input parameters:')
    print(json.dumps(vars(args), indent=2))

    result_path = args.result_path  # pkl_dir in dist_test_with_trans2dupctype.py
    save_path = args.save_path
    class_names = args.class_names
    conf_thresh = args.conf_thresh
    reserve_ori_wlh = args.reserve_ori_wlh
    trans_headstock = args.trans_headstock
    headstock_delta_list = args.headstock_delta
    class_names = class_names.split(',')
    id2class = {i: name.strip() for i, name in enumerate(class_names)}
    process_num = args.process_num

    multiprocessing.set_start_method('spawn')

    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    rotate45degree = False

    start_time = time.time()
    dump_processes = []

    with open(result_path, 'rb') as fp:
        results = pickle.load(fp)
    # print("load from result path: ", args.result_path)
    # results = paddle.load(args.result_path)
    
    # print(results)
    # exit()
    for i in range(process_num):
        process = DumpProcess(i, process_num, results, conf_thresh,
                              rotate45degree, save_path, reserve_ori_wlh, id2class,
                              trans_headstock, headstock_delta_list
                              )
        process.start()
        dump_processes.append(process)

    for i in range(process_num):
        dump_processes[i].join()
    end_time = time.time()
    print("cost time: ", end_time - start_time,
          'finish inference and trans2dupctype')
