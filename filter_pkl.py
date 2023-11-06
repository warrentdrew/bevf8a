import pickle
from copy import copy
import numpy as np

thresh = 10000
pklpath = 'data/hesai90/hesai90_train_20k_noweakly.pkl'

with open(pklpath, 'rb') as f:
    origin_data = pickle.load(f)

# new_data_infos = deepcopy(origin_data)

data_infos = origin_data['infos']
new_data_infos = [] # deepcopy(data_infos)
count = 0

for i, item in enumerate(data_infos):
    cam_infos = item['cams']
    # print(img_meta)
    toadd = True
    for key in cam_infos:
        matrix = cam_infos[key]['lidar2cam_dynamic']
        if np.max(np.abs(matrix)) > thresh:
            toadd = False
            print("cam datapath: ", cam_infos[key]['data_path'])
            print("error matrix: ", matrix)

    if toadd:
        new_data_infos.append(item)
    else:
        count += 1

new_data = {
    'infos' : new_data_infos,
    'metas' : origin_data['metas']
}


print("removed data count: ", count)

# # save pkl
savefile = 'data/hesai90/hesai90_train_20k_noweakly_clean.pkl'
with open(savefile, 'wb') as f:
    pickle.dump(new_data, f)
