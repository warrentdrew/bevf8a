import paddle
import torch
paddle_weight = paddle.load('init.pdparams')
torch_weight = torch.load('weights/torch_init.pth')#['state_dict']

# 过滤num_batches_tracked参数
torch_weight_n = {}
for key in torch_weight.keys():
    if 'num_batches_tracked' in key:
        continue
    torch_weight_n[key] = torch_weight[key]

# 根据参数名称和shape 对比paddle和torch参数
for kt, kp in zip(torch_weight_n.keys(), paddle_weight.keys()):
    print(kt, torch_weight_n[kt].shape, kp, paddle_weight[kp].shape)

# # 开始转参数
from collections import OrderedDict
new_weight_dict = OrderedDict()
for torch_key, paddle_key in zip(torch_weight_n.keys(), paddle_weight.keys()):
    if len(paddle_weight[paddle_key].shape)==2:
        # paddle的fc层的weight与竞品不太一致，需要转置一下
        if ('bev_embedding' in paddle_key) or ('cams_embeds' in paddle_key) or ('level_embeds' in paddle_key):
            new_weight_dict[paddle_key] = torch_weight_n[torch_key].detach().cpu().numpy().astype(paddle_weight[paddle_key].numpy().dtype)
            print('not', paddle_key, torch_weight_n[torch_key].shape, paddle_weight[paddle_key].shape)
        else:
            new_weight_dict[paddle_key] = torch_weight_n[torch_key].detach().cpu().numpy().astype(paddle_weight[paddle_key].numpy().dtype).T
            print(paddle_key, torch_weight_n[torch_key].shape, paddle_weight[paddle_key].shape)
    else:
        new_weight_dict[paddle_key] = torch_weight_n[torch_key].detach().cpu().numpy().astype(paddle_weight[paddle_key].numpy().dtype)
# 保存paddle参数
paddle.save(new_weight_dict, 'weights/torch_init_remapped_wns.pdparams')
print('end save')