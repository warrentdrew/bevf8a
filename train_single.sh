export CUDA_VISIBLE_DEVICES=1
python tools/train.py \
--config configs/bevfusion_idg/fusion/8A/bevfusion_baseline.yaml \
--num_workers 4  \
--save_interval 1 \
--log_interval 5 \
--keep_checkpoint_max 20 \
--save_dir out_bevf_lidarrcnn_stage1_tc \
# --model cvt_weights/maptr_torchinit_dict_qkvreshape_remapped.pdparams