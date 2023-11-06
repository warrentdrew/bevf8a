export CUDA_VISIBLE_DEVICES=6
python tools/train.py \
--config configs/bevfusion_idg/fusion/8A/bevfusion_baseline.yaml \
--num_workers 0  \
--save_interval 1 \
--log_interval 1 \
--keep_checkpoint_max 20 \
--save_dir out_bevf_lidarrcnn_stage1_tc \
--model weights/torch_init_1030_remapped_wns.pdparams