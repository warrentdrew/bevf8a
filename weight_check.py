import torch
import paddle
import numpy as np


torchfile = 'weights/epoch_6_mothead.pth'
paddlefile = 'weights/epoch_6_mothead_remapped.pdparams'
torch_dict = torch.load(torchfile)['state_dict']# .state_dict()
paddle_dict = paddle.load(paddlefile)# .to_dict()
# print(torch_dict)
# reduc_conv.bn.num_batches_tracked
# print(paddle_dict)

fc_names = ['pts_voxel_encoder.pfn_layers.0.0',
            'img_backbone.backbone.stages.0.attn.fc',
            'img_backbone.backbone.stages.1.attn.fc'
            'img_backbone.backbone.stages.2.attn.fc'
            'img_backbone.backbone.stages.3.attn.fc']

for k in torch_dict:
    if k.endswith("num_batches_tracked"):
        continue

    print("key: ", k)
    torch_v = torch_dict[k]
    paddle_k = k.replace("module.", "")
    if paddle_k.endswith('running_mean'):
        paddle_k = paddle_k.replace('running_mean','_mean')
    if paddle_k.endswith('running_var'):
        paddle_k = paddle_k.replace('running_var','_variance')
    paddle_v = paddle_dict[paddle_k]
    torch_v_np = torch_v.detach().cpu().numpy()
    paddle_v_np = paddle_v.numpy()
    flag = [i in k for i in fc_names]
    if any(flag) and "weight" in k:
        paddle_v_np = np.transpose(paddle_v_np, (1, 0))
    np.testing.assert_allclose(torch_v_np, paddle_v_np)

