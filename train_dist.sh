# export GLOG_v=4
# ps -ef | grep 8A/bevfusion_baseline.yaml | grep -v grep | awk '{print "kill -9 "$2}'|sh
export CUDA_VISIBLE_DEVICES=5,6
python -m paddle.distributed.launch tools/train.py \
--config configs/bevfusion_idg/fusion/8A/bevfusion_baseline.yaml \
--num_workers 4  \
--save_interval 1 \
--log_interval 1 \
--keep_checkpoint_max 20 \
--save_dir out_bevf_stage1_exptest \
--model weights/torch_init_1030_remapped_wns.pdparams