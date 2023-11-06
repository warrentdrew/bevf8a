export CUDA_VISIBLE_DEVICES=4
### eval ###
python tools/evaluate.py \
    --config configs/bevfusion_idg/fusion/8A/bevfusion_baseline.yaml \
    --model out_bevf_stage1_exp5_develop/epoch_6/model.pdparams \
    --num_workers 8
    