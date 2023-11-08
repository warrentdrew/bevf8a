/*
Point cloud feature pooling
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <math.h>
#include <stdio.h>
#include <assert.h>

#define THREADS_PER_BLOCK 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define CHECK_CALL(call)                                \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

__device__ inline int pt_in_box3d(float x, float y, float z, float cx, float cy, float cz, float l, float w,
                                  float h, float angle, float max_dis)
{
    float x_rot, y_rot, cosa, sina;
    int in_flag;

    if ((fabsf(x - cx) > max_dis) || (fabsf(y - cy) > max_dis) || (fabsf(z - cz) > h / 2.0))
    {
        return 0;
    }
    cosa = cos(angle);
    sina = sin(angle);
    x_rot = (x - cx) * cosa + (y - cy) * (-sina);
    y_rot = (x - cx) * sina + (y - cy) * cosa;

    in_flag = (x_rot >= -l / 2.0) & (x_rot <= l / 2.0) & (y_rot >= -w / 2.0) & (y_rot <= w / 2.0);
    return in_flag;
}

__global__ void roipool3d_forward(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                                  const float *xyz, const float *boxes3d, const float *pts_feature,
                                  float *pooled_features, int *pooled_empty_flag)
{
    // params xyz: (B, N, 3)
    // params boxes3d: (B, M, 7)
    // params pts_feature: (B, N, C)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)

    int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boxes_idx >= boxes_num)
    {
        return;
    }

    for (int i = 0; i < batch_size; i++)
    {
        int cnt = 0;
        for (int k = 0; k < pts_num; k++)
        {
            int pt_offset = i * pts_num * 3 + k * 3;
            int box_offset = i * boxes_num * 7 + boxes_idx * 7;

            int cur_in_flag = pt_in_box3d(xyz[pt_offset], xyz[pt_offset + 1], xyz[pt_offset + 2], boxes3d[box_offset],
                                          boxes3d[box_offset + 1], boxes3d[box_offset + 2], boxes3d[box_offset + 3],
                                          boxes3d[box_offset + 4], boxes3d[box_offset + 5], boxes3d[box_offset + 6], 30.0);
            if (cur_in_flag)
            {
                if (cnt < sampled_pts_num)
                {
                    int feature_out_offset = i * boxes_num * sampled_pts_num * (3 + feature_in_len) +
                                             boxes_idx * sampled_pts_num * (3 + feature_in_len) +
                                             cnt * (3 + feature_in_len);

                    int feature_in_offset = i * pts_num * feature_in_len + k * feature_in_len;

                    // copy xyz
                    for (int j = 0; j < 3; j++)
                        pooled_features[feature_out_offset + j] = xyz[pt_offset + j];

                    // copy feature
                    for (int j = 0; j < feature_in_len; j++)
                        pooled_features[feature_out_offset + 3 + j] = pts_feature[feature_in_offset + j];

                    cnt++;
                }
                else
                    break;
            }
        }

        if (cnt == 0)
        {
            pooled_empty_flag[i * boxes_num + boxes_idx] = 1;
        }
        else if (cnt < sampled_pts_num)
        {
            // duplicate same points for sampling
            for (int k = cnt; k < sampled_pts_num; k++)
            {
                int duplicate_idx = k % cnt;
                int src_offset = i * boxes_num * sampled_pts_num * (3 + feature_in_len) +
                                 boxes_idx * sampled_pts_num * (3 + feature_in_len) +
                                 duplicate_idx * (3 + feature_in_len);
                int dst_offset = i * boxes_num * sampled_pts_num * (3 + feature_in_len) +
                                 boxes_idx * sampled_pts_num * (3 + feature_in_len) +
                                 k * (3 + feature_in_len);
                for (int j = 0; j < 3 + feature_in_len; j++)
                    pooled_features[dst_offset + j] = pooled_features[src_offset + j];
            }
        }
    }
}

__global__ void assign_pts_to_box3d(int batch_size, int pts_num, int boxes_num, const float *xyz, const float *boxes3d, int *pts_assign)
{
    // params xyz: (B, N, 3)
    // params boxes3d: (B, M, 7)
    // params pts_assign: (B, N, M): idx of the corresponding box3d, -1 means background points
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;

    if (pt_idx >= pts_num || box_idx >= boxes_num || bs_idx >= batch_size)
    {
        return;
    }
    int assign_idx = bs_idx * pts_num * boxes_num + pt_idx * boxes_num + box_idx;
    pts_assign[assign_idx] = 0;

    int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;
    int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;

    int cur_in_flag = pt_in_box3d(xyz[pt_offset], xyz[pt_offset + 1], xyz[pt_offset + 2], boxes3d[box_offset],
                                  boxes3d[box_offset + 1], boxes3d[box_offset + 2], boxes3d[box_offset + 3],
                                  boxes3d[box_offset + 4], boxes3d[box_offset + 5], boxes3d[box_offset + 6], 30.0);

    pts_assign[assign_idx] = cur_in_flag;
    // printf("bs=%d, pt=%d, in=%d\n", bs_idx, pt_idx, pts_assign[bs_idx * pts_num + pt_idx]);
}

__global__ void assign_pts_to_box3dv2(int pts_num, int boxes_num, const float *xyz, const float *boxes3d, char *pts_assign)
{
    // params xyz: (N, 3)
    // params boxes3d: (M, 7)
    // params pts_assign: (M, N): idx of the corresponding box3d, -1 means background points
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x; // 第一维度表示box索引
    int pt_idx = blockIdx.y * blockDim.y + threadIdx.y;  // blockDim.y为512， /第二维表示点索引
    assert(box_idx < boxes_num);
    if (pt_idx >= pts_num)
    {
        return;
    }
    // int assign_idx = pt_idx * boxes_num + box_idx;
    int assign_idx = pt_idx + box_idx * pts_num;
    pts_assign[assign_idx] = 0;

    int box_offset = box_idx * 7;
    int pt_offset = pt_idx * 3;

    int cur_in_flag = pt_in_box3d(xyz[pt_offset], xyz[pt_offset + 1], xyz[pt_offset + 2], boxes3d[box_offset],
                                  boxes3d[box_offset + 1], boxes3d[box_offset + 2], boxes3d[box_offset + 3],
                                  boxes3d[box_offset + 4], boxes3d[box_offset + 5], boxes3d[box_offset + 6], 30.0);

    pts_assign[assign_idx] = (char)cur_in_flag;
    // printf("bs=%d, pt=%d, in=%d\n", bs_idx, pt_idx, pts_assign[bs_idx * pts_num + pt_idx]);
}

__global__ void get_pooled_idx(int batch_size, int pts_num, int boxes_num, int sampled_pts_num,
                               const int *pts_assign, int *pts_idx, int *pooled_empty_flag, int *pooled_padding_flag)
{
    // params xyz: (B, N, 3)
    // params pts_feature: (B, N, C)
    // params pts_assign: (B, N)
    // params pts_idx: (B, M, 512)
    // params pooled_empty_flag: (B, M)
    // params pooled_padding_falg: (B, M, 512)

    int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boxes_idx >= boxes_num)
    {
        return;
    }

    int bs_idx = blockIdx.y;

    int cnt = 0;
    for (int k = 0; k < pts_num; k++)
    {
        if (pts_assign[bs_idx * pts_num * boxes_num + k * boxes_num + boxes_idx])
        {
            if (cnt < sampled_pts_num)
            {
                pts_idx[bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num + cnt] = k;
                cnt++;
            }
            else
                break;
        }
    }

    if (cnt == 0)
    {
        pooled_empty_flag[bs_idx * boxes_num + boxes_idx] = 1;
    }

    else if (cnt < sampled_pts_num)
    {
        // duplicate same points for sampling
        int base_offset = bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num;
        for (int k = cnt; k < sampled_pts_num; k++)
        {
            int duplicate_idx = k % cnt;
            pts_idx[base_offset + k] = pts_idx[base_offset + duplicate_idx];
            pooled_padding_flag[base_offset + k] = 1;
        }
    }
}

__global__ void compute_point_iou(int pts_num, int boxes_num1, int boxes_num2,
                                  const int *pts_assign1, const int *pts_assign2, float *iou3d)
{

    // params pts_assign1: (boxes_num1, pts_num)
    // params pts_assign1: (boxes_num2, pts_num)

    int boxes_idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int boxes_idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    int assign_offset1 = boxes_idx1 * pts_num;
    int assign_offset2 = boxes_idx2 * pts_num;
    int iou_offset = boxes_idx1 * boxes_num2 + boxes_idx2;

    if (boxes_idx1 >= boxes_num1 || boxes_idx2 >= boxes_num2)
    {
        return;
    }

    float cnt = 0;
    float cnt1 = 0;
    float cnt2 = 0;
    for (int k = 0; k < pts_num; k++)
    {
        if (pts_assign1[assign_offset1 + k] > 0 && pts_assign2[assign_offset2 + k] > 0)
        {
            cnt = cnt + 1.0;
            cnt1 = cnt1 + 1.0;
            cnt2 = cnt2 + 1.0;
        }
        else if (pts_assign1[assign_offset1 + k] > 0)
        {
            cnt1 = cnt1 + 1.0;
        }
        else if (pts_assign2[assign_offset2 + k] > 0)
        {
            cnt2 = cnt2 + 1.0;
        }
    }
    float sum = max(1.0, cnt1 + cnt2 - cnt);
    iou3d[iou_offset] = cnt / sum;
    // printf("iou_3d=%f, cnt1=%f, cnt2=%f, cnt=%f, box1=%d, box2=%d, threadx=%d, thready=%d, blockx=%d, blocky=%d\n", iou3d[iou_offset], cnt1, cnt2, cnt, boxes_idx1, boxes_idx2, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}


__global__ void compute_point_counts(int pts_num, int boxes_num1, int boxes_num2, 
                                    const char * pts_assign_mask1, const char *pts_assign_mask2,
                                  int *box1_count, int *box2_count, int *box1_box2_count)
{

    // params pts_assign1: (boxes_num1, pts_num)
    // params pts_assign1: (boxes_num2, pts_num)
    
    int pts_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int boxes_idx1 = blockIdx.y * blockDim.y + threadIdx.y;
    int boxes_idx2 = blockIdx.z * blockDim.z + threadIdx.z;
    
    int box1_box2_count_offset = boxes_idx1 * boxes_num2 + boxes_idx2;

    if (boxes_idx1 >= boxes_num1 || boxes_idx2 >= boxes_num2 || pts_idx >= pts_num)
    {
        return;
    }
    // int box_offset1 = boxes_idx1 * 7;
    // int box_offset2 = boxes_idx2 * 7;
    int mask_offset1 = boxes_idx1 * pts_num + pts_idx;
    int mask_offset2 = boxes_idx2 * pts_num + pts_idx;
    // int pt_offset = pts_idx * 3;

  
    char cur_in_flag1 = pts_assign_mask1[mask_offset1];
    char cur_in_flag2 = pts_assign_mask2[mask_offset2];
    // int cur_in_flag1 = pt_in_box3d(xyz[pt_offset], xyz[pt_offset + 1], xyz[pt_offset + 2], boxes3d1[box_offset1],
    //                               boxes3d1[box_offset1 + 1], boxes3d1[box_offset1 + 2], boxes3d1[box_offset1 + 3],
    //                               boxes3d1[box_offset1 + 4], boxes3d1[box_offset1 + 5], boxes3d1[box_offset1 + 6], 30.0);

    // int cur_in_flag2 = pt_in_box3d(xyz[pt_offset], xyz[pt_offset + 1], xyz[pt_offset + 2], boxes3d2[box_offset2],
    //                               boxes3d2[box_offset2 + 1], boxes3d2[box_offset2 + 2], boxes3d2[box_offset2 + 3],
    //                               boxes3d2[box_offset2 + 4], boxes3d2[box_offset2 + 5], boxes3d2[box_offset2 + 6], 30.0);
    if (cur_in_flag1) {
        int cnt1 = atomicAdd(&box1_count[boxes_idx1], 1);
    }
    if (cur_in_flag2) {
        int cnt2 = atomicAdd(&box2_count[boxes_idx2], 1);
    }
    
    if (cur_in_flag1 && cur_in_flag2) {
        int cnt3 = atomicAdd(&box1_box2_count[box1_box2_count_offset], 1);
    }
}

__global__ void compute_point_iouv2(int boxes_num1, int boxes_num2, const int *box1_count, const int *box2_count, const int *box1_box2_count, float* iou3d)
{

    int boxes_idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int boxes_idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    // int assign_offset1 = boxes_idx1 * pts_num;
    // int assign_offset2 = boxes_idx2 * pts_num;
    int iou_offset = boxes_idx1 * boxes_num2 + boxes_idx2;

    if (boxes_idx1 >= boxes_num1 || boxes_idx2 >= boxes_num2)
    {
        return;
    }
    float sum = float(MAX(1.0, box1_count[boxes_idx1] / boxes_num2 + box2_count[boxes_idx2] / boxes_num1 - box1_box2_count[iou_offset]));
    float cnt = float(box1_box2_count[iou_offset]);

    iou3d[iou_offset] = cnt / sum;
}

__global__ void roipool3d_forward(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                                  const float *xyz, const int *pts_idx, const float *pts_feature,
                                  float *pooled_features, int *pooled_empty_flag)
{
    // params xyz: (B, N, 3)
    // params pts_idx: (B, M, 512)
    // params pts_feature: (B, N, C)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)

    int sample_pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;

    if (sample_pt_idx >= sampled_pts_num || box_idx >= boxes_num || bs_idx >= batch_size)
    {
        return;
    }

    if (pooled_empty_flag[bs_idx * boxes_num + box_idx])
    {
        return;
    }

    int temp_idx = bs_idx * boxes_num * sampled_pts_num + box_idx * sampled_pts_num + sample_pt_idx;
    int src_pt_idx = pts_idx[temp_idx];
    // if (src_pt_idx < 0) {
    //     return;
    // }
    int dst_feature_offset = temp_idx * (3 + feature_in_len);

    for (int j = 0; j < 3; j++)
        pooled_features[dst_feature_offset + j] = xyz[bs_idx * pts_num * 3 + src_pt_idx * 3 + j];

    int src_feature_offset = bs_idx * pts_num * feature_in_len + src_pt_idx * feature_in_len;
    for (int j = 0; j < feature_in_len; j++)
        pooled_features[dst_feature_offset + 3 + j] = pts_feature[src_feature_offset + j];
}

void roipool3dLauncher_slow(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                            const float *xyz, const float *boxes3d, const float *pts_feature, float *pooled_features, int *pooled_empty_flag)
{
    roipool3d_forward<<<DIVUP(boxes_num, THREADS_PER_BLOCK),
                        THREADS_PER_BLOCK>>>(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
                                             xyz, boxes3d, pts_feature, pooled_features, pooled_empty_flag);

#ifdef DEBUG
    cudaDeviceSynchronize(); // for using printf in kernel function
#endif
}

void roipool3dLauncher(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num,
                       const float *xyz, const float *boxes3d, const float *pts_feature, float *pooled_features, int *pooled_empty_flag, int *pooled_padding_flag)
{

    // printf("batch_size=%d, pts_num=%d, boxes_num=%d\n", batch_size, pts_num, boxes_num);
    int *pts_assign = NULL;
    cudaMalloc(&pts_assign, batch_size * pts_num * boxes_num * sizeof(int)); // (batch_size, N, M)
    // cudaMemset(&pts_assign, -1, batch_size * pts_num * boxes_num * sizeof(int));
    // cudaError_t error0 = cudaGetLastError();
    // printf("CUDA error0: %s\n", cudaGetErrorString(error0));
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num, batch_size); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assign_pts_to_box3d<<<blocks, threads>>>(batch_size, pts_num, boxes_num, xyz, boxes3d, pts_assign);
    // cudaError_t error1 = cudaGetLastError();
    // printf("CUDA error1: %s\n", cudaGetErrorString(error1));
    int *pts_idx = NULL;
    cudaMalloc(&pts_idx, batch_size * boxes_num * sampled_pts_num * sizeof(int)); // (batch_size, M, sampled_pts_num)
    // cudaMemset(&pts_idx, -1, batch_size * boxes_num * sampled_pts_num * sizeof(int));
    // cudaError_t error2 = cudaGetLastError();
    // printf("CUDA error2: %s\n", cudaGetErrorString(error2));

    dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK), batch_size); // blockIdx.x(col), blockIdx.y(row)
    get_pooled_idx<<<blocks2, threads>>>(batch_size, pts_num, boxes_num, sampled_pts_num, pts_assign, pts_idx, pooled_empty_flag, pooled_padding_flag);
    // cudaError_t error3 = cudaGetLastError();
    // printf("CUDA error3: %s\n", cudaGetErrorString(error3));

    dim3 blocks_pool(DIVUP(sampled_pts_num, THREADS_PER_BLOCK), boxes_num, batch_size);
    roipool3d_forward<<<blocks_pool, threads>>>(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
                                                xyz, pts_idx, pts_feature, pooled_features, pooled_empty_flag);

    // cudaError_t error4 = cudaGetLastError();
    // printf("CUDA error4: %s\n", cudaGetErrorString(error4));

    cudaFree(pts_assign);
    cudaFree(pts_idx);

#ifdef DEBUG
    cudaDeviceSynchronize(); // for using printf in kernel function
#endif
}



void pointsiouLauncherV2(int pts_num, int boxes_num1, int boxes_num2, const float *pts, const float *boxes3d1, const float *boxes3d2, float *iou3d)
{
    
    char *pts_assign_mask1 = NULL;
    CHECK_CALL(cudaMalloc(&pts_assign_mask1, boxes_num1 * pts_num * sizeof(char))); // (N, M)
    dim3 blocks(boxes_num1, DIVUP(pts_num, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(1, THREADS_PER_BLOCK);
    assign_pts_to_box3dv2<<<blocks, threads>>>(pts_num, boxes_num1, pts, boxes3d1, pts_assign_mask1);

    char *pts_assign_mask2 = NULL;
    CHECK_CALL(cudaMalloc(&pts_assign_mask2, boxes_num2 * pts_num * sizeof(char))); // (batch_size, N, M)
    dim3 blocks2(boxes_num2, DIVUP(pts_num, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    assign_pts_to_box3dv2<<<blocks2, threads>>>(pts_num, boxes_num2, pts, boxes3d2, pts_assign_mask2);
   
    int *box1_pts_count = NULL;
    CHECK_CALL(cudaMalloc(&box1_pts_count, boxes_num1 * sizeof(int))); // (batch_size, N, M)
    CHECK_CALL(cudaMemset(box1_pts_count, 0, boxes_num1 * sizeof(int)));

    int *box2_pts_count = NULL;
    CHECK_CALL(cudaMalloc(&box2_pts_count, boxes_num2 * sizeof(int))); // (batch_size, N, M)
    CHECK_CALL(cudaMemset(box2_pts_count, 0, boxes_num2 * sizeof(int)));

    int *box1_box2_count = NULL;
    CHECK_CALL(cudaMalloc(&box1_box2_count, boxes_num2 * boxes_num1 * sizeof(int))); // (batch_size, N, M)
    CHECK_CALL(cudaMemset(box1_box2_count, 0, boxes_num2 * boxes_num1 * sizeof(int)));

    dim3 blocks1(pts_num, DIVUP(boxes_num1, 32), DIVUP(boxes_num2, 32)); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads1(1, 32, 32);
    compute_point_counts<<<blocks1, threads1>>>(pts_num, boxes_num1, boxes_num2, pts_assign_mask1, pts_assign_mask2, box1_pts_count, box2_pts_count, box1_box2_count);
    
    dim3 blocks3(DIVUP(boxes_num1, 32), DIVUP(boxes_num2, 32)); // blockIdx.x(col), blockIdx.y(row)
    dim3 threads3(32, 32);
    compute_point_iouv2<<<blocks3, threads3>>>(boxes_num1, boxes_num2, box1_pts_count, box2_pts_count, box1_box2_count, iou3d);

    cudaFree(box1_pts_count);
    cudaFree(box2_pts_count);
    cudaFree(box1_box2_count);
    cudaFree(pts_assign_mask1);
    cudaFree(pts_assign_mask2);
    
#ifdef DEBUG
    cudaDeviceSynchronize(); // for using printf in kernel function
#endif
}
