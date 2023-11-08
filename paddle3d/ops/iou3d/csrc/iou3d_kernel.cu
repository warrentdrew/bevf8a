/* 
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2018. 
*/

#include <stdio.h>
#define THREADS_PER_BLOCK 16
#define THREADS_PER_BLOCK_ANCHOR 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

//#define DEBUG
const int THREADS_PER_BLOCK_NMS = sizeof(int64_t) * 8; 
const float EPS = 1e-8;
struct Point {
    float x, y;
    __device__ Point() {}
    __device__ Point(double _x, double _y){
        x = _x, y = _y;
    }

    __device__ void set(float _x, float _y){
        x = _x; y = _y;
    }

    __device__ Point operator +(const Point &b)const{
        return Point(x + b.x, y + b.y);
    }

    __device__ Point operator -(const Point &b)const{
        return Point(x - b.x, y - b.y);
    }
};

__device__ inline float cross(const Point &a, const Point &b){
    return a.x * b.y - a.y * b.x;
}

__device__ inline float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = min(p1.x,p2.x) <= max(q1.x,q2.x)  && 
              min(q1.x,q2.x) <= max(p1.x,p2.x) &&
              min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}

__device__ inline int check_in_box2d(const float *box, const Point &p){
    //params: box (5) [x1, y1, x2, y2, angle]
    const float MARGIN = 1e-5;

    float center_x = (box[0] + box[2]) / 2;
    float center_y = (box[1] + box[3]) / 2;
    float angle_cos = cos(-box[4]), angle_sin = sin(-box[4]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin + center_x;
    float rot_y = -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y;
#ifdef DEBUG
    printf("box: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", box[0], box[1], box[2], box[3], box[4]);
    printf("center: (%.3f, %.3f), cossin(%.3f, %.3f), src(%.3f, %.3f), rot(%.3f, %.3f)\n", center_x, center_y,
            angle_cos, angle_sin, p.x, p.y, rot_x, rot_y);
#endif
    return (rot_x > box[0] - MARGIN && rot_x < box[2] + MARGIN && rot_y > box[1] - MARGIN && rot_y < box[3] + MARGIN);
}

__device__ inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
    // fast exclusion 
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS){
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    
    }
    else{
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }
    
    return 1;
}

__device__ inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x;
    float new_y = -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

__device__ inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

__device__ inline float box_overlap(const float *box_a, const float *box_b){
    // params: box_a (5) [x1, y1, x2, y2, angle]
    // params: box_b (5) [x1, y1, x2, y2, angle]
    
    float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 = box_a[3], a_angle = box_a[4];
    float b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[2], b_y2 = box_b[3], b_angle = box_b[4];

    Point center_a((a_x1 + a_x2) / 2, (a_y1 + a_y2) / 2);
    Point center_b((b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2);
#ifdef DEBUG
    printf("a: (%.3f, %.3f, %.3f, %.3f, %.3f), b: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", a_x1, a_y1, a_x2, a_y2, a_angle,
           b_x1, b_y1, b_x2, b_y2, b_angle);
    printf("center a: (%.3f, %.3f), b: (%.3f, %.3f)\n", center_a.x, center_a.y, center_b.x, center_b.y);
#endif

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners 
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++){
#ifdef DEBUG
        printf("before corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
#ifdef DEBUG
        printf("corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                cnt++;
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++){
        for (int i = 0; i < cnt - j - 1; i++){
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)){
                temp = cross_points[i]; 
                cross_points[i] = cross_points[i + 1]; 
                cross_points[i + 1] = temp;
            }
        }
    }

#ifdef DEBUG
    printf("cnt=%d\n", cnt);
    for (int i = 0; i < cnt; i++){
        printf("All cross point %d: (%.3f, %.3f)\n", i, cross_points[i].x, cross_points[i].y);
    }
#endif

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

__device__ inline float iou_bev(const float *box_a, const float *box_b){
    // params: box_a (5) [x1, y1, x2, y2, angle]
    // params: box_b (5) [x1, y1, x2, y2, angle]
    float sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    float sb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

__global__ void nms_kernel(const int boxes_num, const float nms_overlap_thresh,
                           const float *boxes, int64_t *mask){
    //params: boxes (N, 5) [x1, y1, x2, y2, ry]
    //params: mask (N, N/THREADS_PER_BLOCK_NMS)

    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 5];

    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 5 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 0];
        block_boxes[threadIdx.x * 5 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 1];
        block_boxes[threadIdx.x * 5 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 2];
        block_boxes[threadIdx.x * 5 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 3];
        block_boxes[threadIdx.x * 5 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 4];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const float *cur_box = boxes + cur_box_idx * 5;

        int i = 0;
        int64_t t = 0;
        int start = 0;
        if (row_start == col_start) {
          start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (iou_bev(cur_box, block_boxes + i * 5) > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

__global__ void anchors_mask_kernel(const int anchors_num, const int *anchors, const int w, const bool *voxel_mask, bool *anchors_mask){
    const int idx = blockIdx.x * THREADS_PER_BLOCK_ANCHOR + threadIdx.x;
    
    if (idx >= anchors_num){
        return;
    }
    const int xmin = anchors[idx * 4];
    const int xmax = anchors[idx * 4 + 2];
    const int ymin = anchors[idx * 4 + 1];
    const int ymax = anchors[idx * 4 + 3];
    bool has_valid_voxel = false;
    for (int i = xmin; i <= xmax; i++) {
        for (int j = ymin; j <= ymax; j++) {
            if (voxel_mask[j * w + i]) {
                has_valid_voxel = true;
                anchors_mask[idx] = has_valid_voxel;
                return;
            }
        } 
    }
    anchors_mask[idx] = has_valid_voxel;
}

__global__ void boxes_parsing_kernel(const int box_num, const float *boxes_data, const int w, int *parsing_map_data){
    const int idx = blockIdx.x * THREADS_PER_BLOCK_ANCHOR + threadIdx.x;
    
    if (idx >= box_num){
        return;
    }
    const float x1 = boxes_data[idx * 8];
    const float x2 = boxes_data[idx * 8 + 2];
    const float x3 = boxes_data[idx * 8 + 4];
    const float x4 = boxes_data[idx * 8 + 6];

    const float y1 = boxes_data[idx * 8 + 1];
    const float y2 = boxes_data[idx * 8 + 3];
    const float y3 = boxes_data[idx * 8 + 5];
    const float y4 = boxes_data[idx * 8 + 7];

    const int xmin = (int)(min(min(x1, x2), min(x3, x4)));
    const int xmax = (int)(max(max(x1, x2), max(x3, x4)));
    const int ymin = (int)(min(min(y1, y2), min(y3, y4)));
    const int ymax = (int)(max(max(y1, y2), max(y3, y4)));

    int count = 0;

    for(int i=xmin; i <= xmax; ++i) {
        for(int j=ymin; j <= ymax; ++j) {
            double angl1 = (i + 0.5 - x1) * (x2 - x1) + (j + 0.5 - y1) * (y2 - y1);
            double angl2 = (i + 0.5 - x2) * (x3 - x2) + (j + 0.5 - y2) * (y3 - y2);
            double angl3 = (i + 0.5 - x3) * (x4 - x3) + (j + 0.5 - y3) * (y4 - y3);
            double angl4 = (i + 0.5 - x4) * (x1 - x4) + (j + 0.5 - y4) * (y1 - y4);
            if((angl1 < 0 && angl2 < 0 && angl3 < 0 && angl4 < 0) || 
                (angl1 > 0 && angl2 > 0 && angl3 > 0 && angl4 > 0)) {
                parsing_map_data[j * w + i] = 1;
                count ++;
            }
        }
    }

    if (count == 0) {
        parsing_map_data[(int)(y1 * w + x1)] = 1;
    } 
}

__global__ void boxes_parsing_with_weight_kernel(const int box_num, const float *boxes_data, const int w, float *parsing_map_data){
    const int idx = blockIdx.x * THREADS_PER_BLOCK_ANCHOR + threadIdx.x;
    
    if (idx >= box_num){
        return;
    }
    const float x1 = boxes_data[idx * 8];
    const float x2 = boxes_data[idx * 8 + 2];
    const float x3 = boxes_data[idx * 8 + 4];
    const float x4 = boxes_data[idx * 8 + 6];

    const float y1 = boxes_data[idx * 8 + 1];
    const float y2 = boxes_data[idx * 8 + 3];
    const float y3 = boxes_data[idx * 8 + 5];
    const float y4 = boxes_data[idx * 8 + 7];

    const int xmin = (int)(min(min(x1, x2), min(x3, x4)));
    const int xmax = (int)(max(max(x1, x2), max(x3, x4)));
    const int ymin = (int)(min(min(y1, y2), min(y3, y4)));
    const int ymax = (int)(max(max(y1, y2), max(y3, y4)));

    int count = 0;

    for(int i=xmin; i <= xmax; ++i) {
        for(int j=ymin; j <= ymax; ++j) {
            double angl1 = (i + 0.5 - x1) * (x2 - x1) + (j + 0.5 - y1) * (y2 - y1);
            double angl2 = (i + 0.5 - x2) * (x3 - x2) + (j + 0.5 - y2) * (y3 - y2);
            double angl3 = (i + 0.5 - x3) * (x4 - x3) + (j + 0.5 - y3) * (y4 - y3);
            double angl4 = (i + 0.5 - x4) * (x1 - x4) + (j + 0.5 - y4) * (y1 - y4);
            if((angl1 < 0 && angl2 < 0 && angl3 < 0 && angl4 < 0) || 
                (angl1 > 0 && angl2 > 0 && angl3 > 0 && angl4 > 0)) {
                count ++;
            }
        }
    }

    if(count > 0) {
        for(int i=xmin; i <= xmax; ++i) {
            for(int j=ymin; j <= ymax; ++j) {
                double angl1 = (i + 0.5 - x1) * (x2 - x1) + (j + 0.5 - y1) * (y2 - y1);
                double angl2 = (i + 0.5 - x2) * (x3 - x2) + (j + 0.5 - y2) * (y3 - y2);
                double angl3 = (i + 0.5 - x3) * (x4 - x3) + (j + 0.5 - y3) * (y4 - y3);
                double angl4 = (i + 0.5 - x4) * (x1 - x4) + (j + 0.5 - y4) * (y1 - y4);
                if((angl1 < 0 && angl2 < 0 && angl3 < 0 && angl4 < 0) || 
                    (angl1 > 0 && angl2 > 0 && angl3 > 0 && angl4 > 0)) {
                    parsing_map_data[j * w + i] = max(parsing_map_data[j * w + i], 1.0 / count);
                }
            }
        }
    }

    if (count == 0) {
        parsing_map_data[(int)(y1 * w + x1)] = 1;
    }
}

__global__ void parsing_box_confs_kernel(const int box_num, const float *boxes_data, const int w, const float *conf_map_data, float *confs_data){
    const int idx = blockIdx.x * THREADS_PER_BLOCK_ANCHOR + threadIdx.x;
    if (idx >= box_num){
        return;
    }
    const float x1 = boxes_data[idx * 8];
    const float x2 = boxes_data[idx * 8 + 2];
    const float x3 = boxes_data[idx * 8 + 4];
    const float x4 = boxes_data[idx * 8 + 6];

    const float y1 = boxes_data[idx * 8 + 1];
    const float y2 = boxes_data[idx * 8 + 3];
    const float y3 = boxes_data[idx * 8 + 5];
    const float y4 = boxes_data[idx * 8 + 7];

    const int xmin = (int)(min(min(x1, x2), min(x3, x4)));
    const int xmax = (int)(max(max(x1, x2), max(x3, x4)));
    const int ymin = (int)(min(min(y1, y2), min(y3, y4)));
    const int ymax = (int)(max(max(y1, y2), max(y3, y4)));

    int count = 0;
    float conf_sum = 0;

    for(int i=xmin; i <= xmax; ++i) {
        for(int j=ymin; j <= ymax; ++j) {
            double angl1 = (i + 0.5 - x1) * (x2 - x1) + (j + 0.5 - y1) * (y2 - y1);
            double angl2 = (i + 0.5 - x2) * (x3 - x2) + (j + 0.5 - y2) * (y3 - y2);
            double angl3 = (i + 0.5 - x3) * (x4 - x3) + (j + 0.5 - y3) * (y4 - y3);
            double angl4 = (i + 0.5 - x4) * (x1 - x4) + (j + 0.5 - y4) * (y1 - y4);
            if((angl1 < 0 && angl2 < 0 && angl3 < 0 && angl4 < 0) || 
                (angl1 > 0 && angl2 > 0 && angl3 > 0 && angl4 > 0)) {
                if(conf_map_data[j * w + i] >= 0) {
                    conf_sum += conf_map_data[j * w + i];
                    count ++;
                }
            }
        }
    }

    if (count == 0) {
        if(conf_map_data[(int)(y1 * w + x1)] >= 0) {
            confs_data[idx] = conf_map_data[(int)(y1 * w + x1)];
        } else {
            confs_data[idx] = 0;
        }
    } else {
        confs_data[idx] = conf_sum / count;
    }
}


__device__ inline float iou_normal(float const * const a, float const * const b) {
    float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
    float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
    float width = fmaxf(right - left, 0.f), height = fmaxf(bottom - top, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0]) * (a[3] - a[1]);
    float Sb = (b[2] - b[0]) * (b[3] - b[1]);
    return interS / fmaxf(Sa + Sb - interS, EPS);
}


__global__ void nms_normal_kernel(const int boxes_num, const float nms_overlap_thresh,
                           const float *boxes, int64_t *mask){
    //params: boxes (N, 5) [x1, y1, x2, y2, ry]
    //params: mask (N, N/THREADS_PER_BLOCK_NMS)

    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS, THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 5];

    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 5 + 0] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 0];
        block_boxes[threadIdx.x * 5 + 1] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 1];
        block_boxes[threadIdx.x * 5 + 2] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 2];
        block_boxes[threadIdx.x * 5 + 3] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 3];
        block_boxes[threadIdx.x * 5 + 4] = boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 4];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
        const float *cur_box = boxes + cur_box_idx * 5;

        int i = 0;
        int64_t t = 0;
        int start = 0;
        if (row_start == col_start) {
          start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (iou_normal(cur_box, block_boxes + i * 5) > nms_overlap_thresh){
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
        mask[cur_box_idx * col_blocks + col_start] = t;
    }
}


void NmsLauncher(const cudaStream_t &stream, const float *boxes, int64_t * mask, int boxes_num, float nms_overlap_thresh){
    dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
                DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
    dim3 threads(THREADS_PER_BLOCK_NMS);
    nms_kernel<<<blocks, threads, 0, stream>>>(boxes_num, nms_overlap_thresh, boxes, mask);
}


void NmsNormalLauncher(const cudaStream_t &stream, const float *boxes, int64_t * mask, int boxes_num, float nms_overlap_thresh){
    dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
                DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
    dim3 threads(THREADS_PER_BLOCK_NMS);
    nms_normal_kernel<<<blocks, threads, 0, stream>>>(boxes_num, nms_overlap_thresh, boxes, mask);
}

void AnchorsMaskLauncher(const cudaStream_t &stream, const int anchors_num, const int *anchors, const int w, const bool *voxel_mask, bool *anchors_mask){

    dim3 blocks(DIVUP(anchors_num, THREADS_PER_BLOCK_ANCHOR));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK_ANCHOR);
    // std::cout << " hello2" << std::endl;
    anchors_mask_kernel<<<blocks, threads, 0, stream>>>(anchors_num, anchors, w, voxel_mask, anchors_mask);
}

void BoxesToParsingLauncher(const cudaStream_t &stream, const int box_num, const float *boxes_data, const int w, int *parsing_map_data){
    dim3 blocks(DIVUP(box_num, THREADS_PER_BLOCK_ANCHOR));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK_ANCHOR);
    
    boxes_parsing_kernel<<<blocks, threads, 0, stream>>>(box_num, boxes_data, w, parsing_map_data);
}

void BoxesToParsingWithWeightLauncher(const cudaStream_t &stream, const int box_num, const float *boxes_data, const int w, float *parsing_map_data){
    dim3 blocks(DIVUP(box_num, THREADS_PER_BLOCK_ANCHOR));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK_ANCHOR);
    
    boxes_parsing_with_weight_kernel<<<blocks, threads, 0, stream>>>(box_num, boxes_data, w, parsing_map_data);
}

void ParsingToBoxesConfLauncher(const cudaStream_t &stream, const int box_num, const float *boxes_data, const int w, const float *conf_map_data, float *confs_data){
    dim3 blocks(DIVUP(box_num, THREADS_PER_BLOCK_ANCHOR));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK_ANCHOR);
    
    parsing_box_confs_kernel<<<blocks, threads, 0, stream>>>(box_num, boxes_data, w, conf_map_data, confs_data);
}

