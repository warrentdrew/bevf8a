export CUDA_VISIBLE_DEVICES=6
### eval ###
python tools/evaluate.py \
    --config configs/bevfusion_idg/fusion/8A/bevfusion_baseline.yaml \
    --model out_bevf_stage1_exp6_torchinit_fixroiinput/epoch_6/model.pdparams \
    --num_workers 8