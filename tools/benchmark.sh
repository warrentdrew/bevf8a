#!/bin/bash
# set -e
set -x

work_dir=$1
gt_label=$2
gt_pcd=$3
inference_result=$4
hdmap=$5
conf=$6
distance=120

mkdir -p $work_dir/
benchmark_eval=/code_dev/baidu/adu/obstacle-benchmark-bigmot/build/benchmark/lidar_detection_benchmark
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/code_dev/baidu/adu-3rd/protobuf/lib:${LD_LIBRARY_PATH}
log_path=$work_dir/log.txt
res_path=$work_dir/res.txt
test_set=${work_dir##*/}

if [ -d $hdmap ]; then
	reserve="JACCARD:0.5|RANGE:roi|IGNORE_OUTSIDE_ROI:true|OVERALL_DISTANCE:120|RECALL_DIM:101|LABEL_BLACK_LIST:others,stopBar,smallUnmovable,crashBarrel,safetyBarrier,sign,accessory,OnlyBicycle,accessory_main|BG_REGION_LIST:fog,fp|CONFIDENCE:${conf}|EXPAND_BBOX_WIDTH:0.4|IGNORE_DT_FENCE:true"
    ${benchmark_eval} \
		--cloud=${gt_pcd} \
		--result=${inference_result} \
		--groundtruth=${gt_label} \
		--loading_thread_num=8 \
		--evaluation_thread_num=8 \
		--parallel_processing_num=8 \
		--is_gt_proto=false \
		--is_gt_binary_proto=true \
		--is_result_proto=false \
		--is_result_binary_proto=true \
		--is_result_future_proto=true \
		--is_folder=true \
		--buffer_gt_wl=0 \
		--bigmot_minlength=0.0 \
		--fei_max_roi=60.0 \
		--fei_max_sub_roi=20.0 \
		--is_static=false \
        --hdmap=$hdmap \
		--reserve=$reserve 2>&1 | tee $log_path
else
	reserve="JACCARD:0.5|RANGE:distance|OVERALL_DISTANCE:120|RECALL_DIM:101|LABEL_BLACK_LIST:others,fog,unknow,unknown,stopBar,smallUnmovable,crashBarrel,safetyBarrier,sign,OnlyBicycle,unknownMovable,accessory,accessory_main|CONFIDENCE:${conf}"
    ${benchmark_eval} \
		--cloud=${gt_pcd} \
		--result=${inference_result} \
		--groundtruth=${gt_label} \
		--loading_thread_num=8 \
		--evaluation_thread_num=8 \
		--parallel_processing_num=8 \
		--is_gt_proto=false \
		--is_gt_binary_proto=true \
		--is_result_proto=false \
		--is_result_binary_proto=true \
		--is_result_future_proto=true \
		--is_folder=true \
		--buffer_gt_wl=0 \
		--bigmot_minlength=0.0 \
		--fei_max_roi=60.0 \
		--fei_max_sub_roi=20.0 \
		--is_static=false \
		--reserve=$reserve 2>&1 | tee $log_path
fi

python ./tools/scripts/collect_eval_info.py ${log_path} ${res_path} ${test_set}