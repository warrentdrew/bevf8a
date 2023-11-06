work_dir='output'
test_set='beijing'
save_model='work_dirs/epoch_6_mothead.pth'
conf=0.0
data_root='data/hesai90'
gpu_num=8
config='configs/online-config/8A/bevfusion_baseline.py'

val_pkl="${data_root}/hesai90_test_5013_beijing.pkl"
gt_pcd=${data_root}/20220818_test_yizhuang_5k/yizhuang_5k/pcd
gt_label=${data_root}/20220818_test_yizhuang_5k/yizhuang_5k/gt_txt_less3_wo_0
hdmap_path=${data_root}/20220818_test_yizhuang_5k/hdmap


export CUDA_VISIBLE_DEVICES=4


## step1 ##
### eval ###
python tools/evaluate.py \
    --config configs/bevfusion_idg/fusion/8A/bevfusion_baseline.yaml \
    --model weights/epoch_6_mothead_remapped.pdparams \
    --num_workers 8


## step2 ##
# python ./tools/trans2dupc_parallel.py \
#         --result_path test_results_1016_convert.pkl \
#         --save_path output/save_dupc_results_1016


## step3 in guojun docker ##
# bash ./tools/benchmark.sh $work_dir $gt_label $gt_pcd $work_dir/save_dupc_results  $hdmap_path $conf

