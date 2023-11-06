# !/usr/bin/env python3
import os
import sys

bigmot_set = {
    'big_mot_val': 69278,
    'shanghai': 30167,
    'beijing': 28064,
    'wuhan_val': 34316,
}


def collect_bigmot(lines: list, total_bigmot_count):
    i =max( len(lines) - 20000, 0)

    bigmot_over_seg = 0
    all_ap = 0
    recall = 0
    precsion = 0
    bigmot_ap = 0
    bigmot_aad = 0
    over_seg_rate = 0
    fei_avg = 0
    fei_95 = 0
    aps = []
    orientation_reverse = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("** over-segmented gt num: "):
            bigmot_over_seg = lines[i + 2].split(':')[-1].strip()
        elif line.startswith("2017 Detection KPI"):
            recall = '%.4f%%' % (float(lines[i + 1].strip().split(' ')[-1]) * 100)
            precsion = '%.4f%%' % (float(lines[i + 2].strip().split(' ')[-1]) * 100)
        elif line.startswith('others ap '):
            aps.append(
                '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
            )
        elif line.startswith('pedestrian ap '):
            aps.append(
                '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
            )
        elif line.startswith('cyclist ap '):
            aps.append(
                '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
            )
        elif line.startswith('vehicle ap '):
            aps.append(
                '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
            )
        elif line.startswith('bigmot ap '):
            bigmot_ap = '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
        elif line.startswith('ap '):
            all_ap = '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
        elif line.startswith("each range AAD"):
            bigmot_aad = lines[i + 6].split(' ')[12]
            i += 5
        elif line.startswith('FittingError'):
            fei_avg = lines[i + 1].split(' ')[12]
            fei_95 = lines[i + 4].strip().split(' ')[-6]
        elif line.startswith('OrientationReverse'):
            orientation_reverse.append('%.2f' % (float(lines[i + 2].split(' ')[17]) * 100))
            orientation_reverse.append('%.2f' % (float(lines[i + 3].split(' ')[17]) * 100))
            orientation_reverse.append('%.2f' % (float(lines[i + 4].split(' ')[18]) * 100))
        i += 1
    out = ' '.join([
        all_ap,
        '%s/%s' % (recall, precsion),
        bigmot_ap,
        bigmot_aad,
        bigmot_over_seg,
        '%.4f%%' % (int(bigmot_over_seg) / total_bigmot_count * 100),
        fei_avg,
        fei_95,
        '|'.join(aps),
        '|'.join(orientation_reverse)
    ])
    return out


def collect_normal(lines: list):
    i = len(lines) - 10000

    bigmot_over_seg = '\\'
    all_ap = 0
    recall = 0
    precsion = 0
    bigmot_ap = '\\'
    bigmot_aad = '\\'
    over_seg_rate = '\\'
    fei_avg = '\\'
    fei_95 = '\\'
    aps = []
    total_bigmot_count = 69278

    while i < len(lines):
        # print("@@len: ", len(lines))
        # print("line1: ", lines[i])
        line = lines[i]
        if line.startswith("2017 Detection KPI"):
            recall = '%.4f%%' % (float(lines[i + 1].strip().split(' ')[-1]) * 100)
            precsion = '%.4f%%' % (float(lines[i + 2].strip().split(' ')[-1]) * 100)
        elif line.startswith('others ap '):
            aps.append(
                '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
            )
        elif line.startswith('pedestrian ap '):
            aps.append(
                '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
            )
        elif line.startswith('cyclist ap '):
            aps.append(
                '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
            )
        elif line.startswith('vehicle ap '):
            aps.append(
                '%.2f' % (float(line.strip().split(' ')[-1]) * 100)
            )
        elif line.startswith('ap '):
            all_ap = '%.4f%%' % (float(line.strip().split(' ')[-1]) * 100)
        i += 1
    out = ' '.join([
        all_ap,
        '%s/%s' % (recall, precsion),
        bigmot_ap,
        bigmot_aad,
        bigmot_over_seg,
        '\\',
        fei_avg,
        fei_95,
        '|'.join(aps)
    ])
    return out


if __name__ == '__main__':
    path = sys.argv[1]
    output = sys.argv[2]
    test_set = sys.argv[3]
    if not output.endswith('txt'):
        output = path.replace('eval_', '')+'.txt'

    # path='/root/paddlejob/workspace/env_run/afs-mount/dingguangyao01/lidar_jobs/dingguangyao01/job-0bb63f0907518406/rank-00000/bigmot2/90p/beijing/eval_120m_6_5w_bcpt20_bigmot_head_120m_dist_0.15'
    # output='./tmp/tmp.txt'
    is_bigmot = test_set in bigmot_set

    with open(path, 'r') as f:
        lines = list(f.readlines())

    if is_bigmot:
        out = collect_bigmot(lines, bigmot_set[test_set])
    else:
        out = collect_normal(lines)

    print(out)
    with open(output, 'a+') as f:
        f.write('\n %s' % out)