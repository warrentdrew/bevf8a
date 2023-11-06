work_dir='output'
test_set='big_mot_val' #'beijing'
# save_model='work_dirs/epoch_6_mothead.pth'
conf=0.0
data_root='data/hesai90'
gpu_num=8
config='configs/online-config/8A/bevfusion_baseline.py'

val_pkl="${data_root}/hesai90_test_bigMot_5633.pkl"
gt_pcd=${data_root}/${test_set}/pcd
gt_label=${data_root}/${test_set}/gt_txt_less3_wo_0/
hdmap_path=${data_root}/${test_set}/hdmap

 # 'work_dirs/epoch_6_mothead.pth' \
### eval ###
# python -m torch.distributed.launch --master_port=50094 --nproc_per_node 7 \
#         ./tools/test.py 'configs/online-config/8A/bevfusion_baseline.py' 'output_check_stage1/epoch_6.pth' \
#         --work-dir 'output' \
#         --val-path 'data/hesai90/hesai90_test_bigMot_5633.pkl' \
#         --out 'output/big_mot_val_result.pkl' \
#         --launcher pytorch



python ./tools/trans2dupc_parallel.py \
        --result_path=test_results_exp5_dev_1031_convert.pkl \
        --save_path=output/save_dupc_results_bigmot_1031_dev \
        --trans_headstock \
        --headstock_delta 0 7

# bash ./tools/benchmark.sh $work_dir $gt_label $gt_pcd $work_dir/save_dupc_results_bigmot_1031_dev  $hdmap_path $conf


