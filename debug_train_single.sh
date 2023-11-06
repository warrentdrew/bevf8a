export CUDA_VISIBLE_DEVICES=0
python -m debugpy --listen 9457 --wait-for-client tools/train.py \
--config configs/bevfusion_idg/fusion/8A/bevfusion_baseline.yaml \
--num_workers 0  \
--save_interval 1 \
--log_interval 5 \
--keep_checkpoint_max 20 \
--save_dir out_bevf_lidarrcnn_stage1_t2 \
--model weights/torch_init_remapped_wns.pdparams #weights/epoch_6_mothead_script_wn.pdparams

