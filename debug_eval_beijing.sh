export CUDA_VISIBLE_DEVICES=4
### eval ###
python -m debugpy --listen 9457 --wait-for-client tools/evaluate.py \
    --config configs/bevfusion_idg/fusion/8A/bevfusion_baseline.yaml \
    --model weights/epoch_6_mothead_script_wn.pdparams \
    --num_workers 0
    