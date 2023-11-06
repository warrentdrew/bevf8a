import pickle
import numpy as np
from copy import deepcopy


def inproj2qkv(torch_param_grads):
    new_torch_param_grads = deepcopy(torch_param_grads)
    for k in torch_param_grads.keys():
        if k.__contains__("in_proj_bias"):
            del new_torch_param_grads[k]
            v = torch_param_grads[k]
            qkv_w = np.split(v, 3)
            q_key_w = k.replace('in_proj_bias', 'q_proj.bias')
            k_key_w = k.replace('in_proj_bias', 'k_proj.bias')
            v_key_w = k.replace('in_proj_bias', 'v_proj.bias')
            new_torch_param_grads[q_key_w] = qkv_w[0]
            new_torch_param_grads[k_key_w] = qkv_w[1]
            new_torch_param_grads[v_key_w] = qkv_w[2]
        if k.__contains__("in_proj_weight"):
            del new_torch_param_grads[k]
            v = torch_param_grads[k]
            qkv_b = np.split(v, 3)
            q_key_b = k.replace('in_proj_weight', 'q_proj.weight')
            k_key_b = k.replace('in_proj_weight', 'k_proj.weight')
            v_key_b = k.replace('in_proj_weight', 'v_proj.weight')
            new_torch_param_grads[q_key_b] = qkv_b[0]
            new_torch_param_grads[k_key_b] = qkv_b[1]
            new_torch_param_grads[v_key_b] = qkv_b[2]
    return new_torch_param_grads


torch_grad_path = '/mnt/zhuyipin/idg/lidarrcnn/BEVFusion/pth_grad_1030.pkl'

pd_grad_path = 'pd_grad_1030.pkl'

with open(torch_grad_path, 'rb') as f:
    torch_grad = pickle.load(f)

with open(pd_grad_path, 'rb') as f:
    pd_grad = pickle.load(f)
# print(pd_grad['grad_info'])
# exit()
torch_param_grads = torch_grad['grad_info']
pd_param_grads = pd_grad['grad_info']
torch_buffers = torch_grad['buffer_info']
pd_buffers = pd_grad['buffer_info']

torch_param_grads_qkv = inproj2qkv(torch_param_grads)

torch_grad_qkv_keys = torch_param_grads_qkv.keys() # [x.replace("module.", "") for x in torch_param_grads_pkv.keys()]
pd_grad_keys = pd_grad['grad_info'].keys()

print("torch grad param: ", len(torch_grad_qkv_keys))
print("paddle grad params: ", len(pd_grad_keys))
# pd_extra_keys = set(pd_grad_keys) - set(torch_grad_keys)
# torch_extra_keys = set(torch_grad_keys) - set(pd_grad_keys)
# print(pd_extra_keys)
# exit()
# print(torch_grad_qkv_keys)


assert len(torch_grad_qkv_keys) == len(pd_grad_keys)
print(len(torch_grad_qkv_keys))
max_diff = 0
max_diff_relative = 0
for i, key in enumerate(torch_grad_qkv_keys):
    print(i, " ", key)
    torch_g = torch_param_grads_qkv[key]

    paddle_key = key.replace("module.", "")
    pd_g = pd_param_grads[paddle_key]

    # =======================================
    # check zero grad
    # print("torch 0 grad: ", len(np.where(torch_g == 0)[0]))
    # print("paddle 0 grad: ",len(np.where(pd_g == 0)[0]))
    # =======================================

    # np.testing.assert_allclose(torch_g, pd_g, atol = 4e-4)
    # if len(pd_g.shape) == 2 and ('embed' not in key) and ("grid_offsets" not in key): #and ("k_proj" not in key) and ("q_proj" not in key) and ("v_proj" not in key) and ("grid_offsets" not in key):
    #     pd_g = pd_g.T

    if len(pd_g.shape) == 2 and not (('bev_embedding' in paddle_key) or ('cams_embeds' in paddle_key) or ('level_embeds' in paddle_key)):
        pd_g = pd_g.T

    diff = np.max(np.abs(torch_g - pd_g))
    relative_diff = np.max(np.abs((torch_g - pd_g) / (torch_g + 1e-12)))
    if max_diff < diff:
        max_diff = diff
        max_diff_key = key
        maxdiff_id = i
    
    if max_diff_relative < relative_diff:
        max_diff_relative = relative_diff
        max_relative_diff_key = key
        max_relative_diff_id = i

    print(f"key: {key}, grad_diff: {diff}, relative_diff: {relative_diff}")
    # if i == 287:
    #     print("torch g", torch_g)
    #     print("=" * 100)
    #     print("paddle g", pd_g)
    #     np.save("torch_grad.npy", torch_g)
    #     np.save("pd_grad.npy", pd_g)

print("max_ diff", max_diff)
print("max_diff_key:", max_diff_key)
print("max diff layer id", maxdiff_id)
print("max_relative_diff_key:", max_relative_diff_key)
print("max relative diff layer id", max_relative_diff_id)