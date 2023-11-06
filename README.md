# 使用文档

## 环境要求

- PaddlePaddle develop（develop训练会快一些，建议使用develop，2.5发布后可以使用2.5稳定版本）
- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64位版本
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 11.0
- cuDNN >= 7.6

## 安装说明

### 1. 安装PaddlePaddle

```
# CUDA11.2
python -m pip install paddlepaddle-gpu==2.4.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
- 更多CUDA版本或环境快速安装，请参考[PaddlePaddle快速安装文档](https://www.paddlepaddle.org.cn/install/quick)
- 更多安装方式例如conda或源码编译安装方法，请参考[PaddlePaddle安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)

请确保您的PaddlePaddle安装成功。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```
**注意**
1. 如果您希望在多卡环境下使用Paddle3D，请首先安装NCCL

### 3. 安装paddle3d

```
# 安装依赖
cd bevfusion
pip install -r requirements.txt

# 编译安装paddle3d
python setup.py develop

# 安装依赖
cd bevfusion/paddle3d/ops
python setup.py install
```

## 使用说明

bevfusion模型的配置文件在`configs/bevfusion_idg/`目录下
数据集与torch的数据集格式和配置完全相同

第一版代码的预训练模型下载

camera模块预训练模型下载地址：https://ecloud.baidu.com?t=38758dbf5106ad47bc9166a41b511909
lidar模块预训练模型下载地址：https://ecloud.baidu.com?t=b687a92c5afc7b33644978ec38caf677
fusion模块预训练模型下载地址：https://ecloud.baidu.com?t=721636a61d1129088f1cd0a26c0deb4f

第二版代码的预训练模型下载

fusion模块预训练模型下载地址：https://ecloud.baidu.com?t=37d4d77cd35b39b5c733a780d5dcd19c

### 训练
单卡训练
```
export CUDA_VISIBLE_DEVICES=0

# cam
python -u tools/train.py --config configs/bevfusion_idg/cam_stream/bevf_pp_1x8_2x_at128_52K_ms022_cam_uniform_grid_transformer_dla34_detrhead_120m_fpnc_bev144x144.yaml --save_dir ./outputs/bevf_cam --save_interval 1 --log_interval 5

# lidar
python -u tools/train.py --config configs/bevfusion_idg/lidar_stream/hv_dynamic_pointpillars_hrnet-multigrouphead_4x8_40ep_at128-3d_120m.yaml --save_dir ./outputs/bevf_lidar --save_interval 1 --log_interval 5

# bevfusion
python -u tools/train.py --config configs/bevfusion_idg/fusion/bevf_ppdet_dla34_uniform-trans_detrhead_2x8_1x_at128_52k_120m_lid288-pretrain52k_cam144-enc1-pretrain52k-ms-cam04-augv1-MGHeadv1.yaml --save_dir ./outputs/bevf_fusion --save_interval 1 --log_interval 5

```

多卡训练
```
# cam
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/bevfusion_idg/cam_stream/bevf_pp_1x8_2x_at128_52K_ms022_cam_uniform_grid_transformer_dla34_detrhead_120m_fpnc_bev144x144.yaml --num_workers 6 --save_dir ./outputs/bevf_cam --save_interval 1 --log_interval 50

# lidar
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/bevfusion_idg/lidar_stream/hv_dynamic_pointpillars_hrnet-multigrouphead_4x8_40ep_at128-3d_120m.yaml --num_workers 6 --save_dir ./outputs/bevf_lidar --save_interval 1 --log_interval 50

# bevfusion
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/bevfusion_idg/fusion/bevf_ppdet_dla34_uniform-trans_detrhead_2x8_1x_at128_52k_120m_lid288-pretrain52k_cam144-enc1-pretrain52k-ms-cam04-augv1-MGHeadv1.yaml --num_workers 6 --save_dir ./outputs/bevf_fusion --save_interval 1 --log_interval 50
```

### 评估
```
export CUDA_VISIBLE_DEVICES=0

# cam
python tools/evaluate.py --config configs/bevfusion_idg/cam_stream/bevf_pp_1x8_2x_at128_52K_ms022_cam_uniform_grid_transformer_dla34_detrhead_120m_fpnc_bev144x144.yaml --model ./outputs/bevf_cam/epoch_24/model.pdparams --num_workers 2

# lidar
python tools/evaluate.py --config configs/bevfusion_idg/lidar_stream/hv_dynamic_pointpillars_hrnet-multigrouphead_4x8_40ep_at128-3d_120m.yaml --model ./outputs/bevf_lidar/epoch_6/model.pdparams --num_workers 2

# bevfusion
python tools/evaluate.py --config configs/bevfusion_idg/fusion/bevf_ppdet_dla34_uniform-trans_detrhead_2x8_1x_at128_52k_120m_lid288-pretrain52k_cam144-enc1-pretrain52k-ms-cam04-augv1-MGHeadv1.yaml --model ./outputs/bevf_fusion/epoch_40/model.pdparams --num_workers 2
```

## 复现结果
训练环境：32G V100 cuda11.2 py3.8 torch1.7.0  paddle2.4.2

### 第一版代码复现结果

cam
复现结果对比 
| 模型 |  cam_easy bmAP | cam_easy mAP | cam_hard bmAP |cam_hard mAP | 
| :----------: | :-------: | :--------: | :------: | :-----: | 
| torch pretrained |  0.5227 | 0.5010 | 0.4294 | 0.4086 | 
| torch trained | 0.5211 | 0.4980 | 0.4287 | 0.4069 | 
| paddle trained |  0.5275 | 0.5050 | 0.4297 | 0.4092 | 

lidar
复现结果对比 
| 模型 |  cam_easy bmAP | cam_easy mAP | cam_hard bmAP |cam_hard mAP | 
| :----------: | :-------: | :--------: | :------: | :-----: | 
| torch pretrained | 0.8626 | 0.8181 | 0.6958 | 0.6728 | 
| torch trained | 0.8571 | 0.8195 | 0.6940 | 0.6708 | 
| paddle trained | 0.8644 | 0.8198 | 0.6989 | 0.6696 | 

fusion
复现结果对比 
| 模型 |  cam_easy bmAP | cam_easy mAP | cam_hard bmAP |cam_hard mAP | 
| :----------: | :-------: | :--------: | :------: | :-----: | 
| torch pretrained |  0.8572 | 0.8195 | 0.6920 | 0.6689 | 
| torch trained | 0.8405 | 0.8020 | 0.6807 | 0.6568 | 
| paddle trained |  0.8496 | 0.8103 | 0.6894 | 0.6648 | 


### 第二版代码复现结果

lidar
复现结果对比 
| 模型 |  cam_easy bmAP | cam_easy mAP | cam_hard bmAP |cam_hard mAP | s/iter |
| :----------: | :-------: | :--------: | :------: | :-----: | :-----: | 
| torch trained | 0.8565 | 0.8197 | 0.6680 | 0.6512 | 0.65 | 
| paddle trained | 0.8561 | 0.8191 | 0.6692 | 0.6521 | 0.65 | 

fusion
复现结果对比 
| 模型 |  cam_easy bmAP | cam_easy mAP | cam_hard bmAP |cam_hard mAP | s/iter/s |
| :----------: | :-------: | :--------: | :------: | :-----: | :-----: | 
| torch trained | 0.8562 | 0.8191 | 0.7079 | 0.6844 | 1.13 | 
| paddle trained |  0.8265 | 0.8194 | 0.7095 | 0.6860 | 1.10 | 
