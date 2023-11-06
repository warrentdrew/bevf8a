function mount_afs_data() {
    afs_name=$1
    username=$2
    password=$3
    local=$4
    remote=$5
    mkdir -p "$local"
    pushd /root/paddlejob/workspace/env_run/afs_mount && \
        nohup ./bin/afs_mount --username="$username" --password="$password" \
            "$local" "${afs_name}${remote}" 1>my_mount.log 2>&1 &
    sleep 2s
    echo "mount $local"
    tree -L 2 "$local"
}

function rand(){
    min=$1
    max=$(($2 - $min + 1))
    num=$(($RANDOM + 1000000000))
    echo $(($num % $max + $min))
}



# ADT image & label
echo "begin mount afs data...."

if ! [ -d afs-mount/ppdet_common/point_cloud/90P_BEV ]; then
    afs_name="afs://xingtian.afs.baidu.com:9902"
    username="ADU_KM_Data"
    password="ADU_km_2018"
    local="/root/paddlejob/workspace/env_run/afs_mount_lidarrcnn/ADU_KM_Data"
    remote="/user/ADU_KM_Data"
    mount_afs_data ${afs_name} ${username} ${password} ${local} ${remote}
fi

fs_name="afs://cygnus.afs.baidu.com:9902"
username="ad-lable-read"
password="ad-lable-zODiYlWe4"
afs_local_mount_point_02="/root/paddlejob/workspace/env_run/afs_mount_lidarrcnn/lable-data"
afs_remote_mount_point_02="/user/adt-platform/lable"
mount_afs_data ${fs_name} ${username} ${password} ${afs_local_mount_point_02} ${afs_remote_mount_point_02}

afs_name="afs://cygnus.afs.baidu.com:9902"
username="ad-perception"
password="ad-perception-c1cA16d8"
local="/root/paddlejob/workspace/env_run/afs_mount_lidarrcnn/heye"
remote="/user/ad-perception/dueye/heye"
mount_afs_data ${afs_name} ${username} ${password} ${local} ${remote}

# # ######### Afs Mount by Hand (ADT90p) ########
fs_name="afs://cnw-xa-main.afs.baidu.com:9902"
username="labeling-w"
password="labeling-w_passw0rd"
afs_remote_mount_point_03="/user/labeling-w/aicv_labeling_data//ADT-perception-train-data/3d-pointcloud-true90-IDA/imagine/tasks"
afs_local_mount_point_03="/root/paddlejob/workspace/env_run/afs_mount_lidarrcnn/ADT90p"
mount_afs_data ${fs_name} ${username} ${password} ${afs_local_mount_point_03} ${afs_remote_mount_point_03}


fs_name="afs://cnw-xa-main.afs.baidu.com:9902"
username="labeling-w"
password="labeling-w_passw0rd"
afs_remote_mount_point_03="/user/labeling-w/aicv_labeling_data//ADT-perception-train-data/"
afs_local_mount_point_03="/root/paddlejob/workspace/env_run/afs_mount_lidarrcnn/ADT-perception-train-data/"
mount_afs_data ${fs_name} ${username} ${password} ${afs_local_mount_point_03} ${afs_remote_mount_point_03}

fs_name="afs://cnw-xa-main.afs.baidu.com:9902"
username="labeling-w"
password="labeling-w_passw0rd"
afs_remote_mount_point_03="/user/labeling-w/atd-vis-rawdata"
afs_local_mount_point_03="/root/paddlejob/workspace/env_run/afs_mount_lidarrcnn/atd-vis-rawdata"
mount_afs_data ${fs_name} ${username} ${password} ${afs_local_mount_point_03} ${afs_remote_mount_point_03}