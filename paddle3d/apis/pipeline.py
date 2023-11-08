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

import copy
import time
import os.path as osp
from tqdm import tqdm
import shutil
import tempfile

import paddle
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.hybrid_parallel_util import \
    fused_allreduce_gradients
from paddle.jit import to_static
import paddle.distributed as dist

from paddle3d.sample import Sample
from paddle3d.utils.logger import logger
from paddle3d.utils.tensor_fusion_utils import all_reduce_parameters
from paddle3d.apis.logger import ProgressBar
from paddle3d.datasets.at128.core import dump, load, mkdir_or_exist

def parse_losses(losses):
    """
    Parse the loss tensor in dictionary into a single scalar.
    """
    log_loss = dict()
    if isinstance(losses, paddle.Tensor):
        total_loss = losses
    elif isinstance(losses, dict):
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, paddle.Tensor):
                log_loss[loss_name] = paddle.sum(loss_value)
            else:
                log_loss[loss_name] = sum(loss_value)
        total_loss = sum(
            _loss_value for _loss_name, _loss_value in log_loss.items())

    log_loss['total_loss'] = total_loss

    return total_loss, log_loss

def training_step(model: paddle.nn.Layer,
                  optimizer: paddle.optimizer.Optimizer,
                  sample: Sample,
                  cur_iter: int,
                  ori_model: paddle.nn.Layer,
                  scaler=None,
                  amp_cfg=dict(),
                  all_fused_tensors=None,
                  group=None,
                  revert_syncbn_status=False) -> dict:

    if optimizer.__class__.__name__ == 'OneCycleAdam':
        optimizer.before_iter(cur_iter - 1)

    model.train()
    if isinstance(model, paddle.DataParallel):
        model._layers.train()
    if revert_syncbn_status:
        if isinstance(model, paddle.DataParallel):
            for ori_mod, mod in zip(ori_model.sublayers(), model._layers.sublayers()):
                if isinstance(ori_mod, (nn.BatchNorm2D, nn.BatchNorm1D)):
                    mod.weight.stop_gradient = ori_mod.weight.stop_gradient
                    mod.bias.stop_gradient = ori_mod.bias.stop_gradient
                    mod.train() if ori_mod.training else mod.eval() 
        else:
            for ori_mod, mod in zip(ori_model.sublayers(), model.sublayers()):
                if isinstance(ori_mod, (nn.BatchNorm2D, nn.BatchNorm1D)):
                    mod.weight.stop_gradient = ori_mod.weight.stop_gradient
                    mod.bias.stop_gradient = ori_mod.bias.stop_gradient
                    mod.train() if ori_mod.training else mod.eval() 
    del ori_model


    if isinstance(model, paddle.DataParallel):
        outputs = model(sample)
        if hasattr(model._layers, '_parse_losses'):
            loss, log_loss = model._layers._parse_losses(outputs['loss'])
        else:
            loss, log_loss = parse_losses(outputs['loss'])
        loss.backward()
    else:
        if scaler is not None:
            with paddle.amp.auto_cast(**amp_cfg):
                outputs = model(sample)
                if hasattr(model, '_parse_losses'):
                    loss, log_loss = model._parse_losses(outputs['loss'])
                else:
                    loss, log_loss = parse_losses(outputs['loss'])
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
        else:
            outputs = model(sample)
            if hasattr(model, '_parse_losses'):
                loss, log_loss = model._parse_losses(outputs['loss'])
            else:
                loss, log_loss = parse_losses(outputs['loss'])
            loss.backward()

    # update params
    if optimizer.__class__.__name__ == 'OneCycleAdam':
        optimizer.after_iter()
    else:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.clear_grad()

        if isinstance(optimizer._learning_rate,
                      paddle.optimizer.lr.LRScheduler):
            optimizer._learning_rate.step()

    # reduce loss when distributed training
    if paddle.distributed.is_initialized() and (not hasattr(model._layers, '_parse_losses')):
        with paddle.no_grad():
            for loss_name, loss_value in log_loss.items():
                loss_clone = loss_value.clone()
                paddle.distributed.all_reduce(
                    loss_clone.scale_(1. / paddle.distributed.get_world_size()))
                log_loss[loss_name] = loss_clone.item()

    return log_loss


def validation_step(model: paddle.nn.Layer, sample: Sample) -> dict:
    model.eval()
    with paddle.no_grad():
        outputs = model(sample)
    return outputs


def apply_to_static(support_to_static, model, image_shape=None):
    if support_to_static:
        specs = None
        if image_shape is not None:
            specs = image_shape
        model = to_static(model, input_spec=specs)
        logger.info(
            "Successfully to apply @to_static with specs: {}".format(specs))
    return model

def single_gpu_test(model, data_loader):
    results = []
    for idx, sample in enumerate(tqdm(data_loader)):
        sample_idx = int(sample.pop('sample_idx')[0])
        filename = data_loader.dataset.data_infos[sample_idx]['cams'][data_loader.dataset.cam_orders[0]]['data_path']
        result = validation_step(model, sample)
        assert len(result) == 1
        result[0]['sample_idx'] = sample_idx
        result[0]['filename'] = filename
        results.extend(result)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    """Test model with multiple gpus."""
    results = []
    dataset = data_loader.dataset
    curr_num, total_num = 0, len(dataset)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    if rank == 0:
        prog_bar = ProgressBar(logger, total_num)

    time.sleep(2)
    for idx, sample in enumerate(data_loader):
        sample_idx = int(sample.pop('sample_idx')[0])
        filename = dataset.data_infos[sample_idx]['cams'][dataset.cam_orders[0]]['data_path']
        result = validation_step(model, sample)
        assert len(result) == 1
        result[0]['sample_idx'] = sample_idx
        result[0]['filename'] = filename
        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            curr_num += batch_size * world_size
            prog_bar.update(curr_num)
    return results



def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = paddle.full((MAX_LEN,), 32, dtype=paddle.uint8)
        if rank == 0:
            mkdir_or_exist(".dist_test")
            tmpdir = tempfile.mkdtemp(dir=".dist_test")
            tmpdir = paddle.to_tensor(bytearray(tmpdir.encode()), dtype=paddle.uint8)
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    dump(result_part, osp.join(tmpdir, f"part_{rank}.pkl"))

    # synchronize all processes
    dist.barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f"part_{i}.pkl")
            part_list.append(load(part_file))

        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]

        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

