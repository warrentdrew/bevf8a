import numpy as np
import torch
import paddle

def torch2paddle():
    torch_path = "weights/epoch_6_mothead.pth"  #"resnet50-0676ba61.pth"
    paddle_path = "weights/epoch_6_mothead_correct.pdparams"
    torch_state_dict = torch.load(torch_path)['state_dict'] #.state_dict() # ['state_dict']
    print(torch_state_dict.keys())
    fc_names = ['pts_voxel_encoder.pfn_layers.0.0',
    'cam2bev_modules.transformer.encoder.layers.0.attentions.0.deformable_attention.sampling_offsets.weight',
    'cam2bev_modules.transformer.encoder.layers.0.attentions.0.deformable_attention.attention_weights.weight',
    'cam2bev_modules.transformer.encoder.layers.0.ffns.0.layers.0.0.weight',
    'cam2bev_modules.transformer.encoder.layers.0.ffns.0.layers.1.weight']
            #     'img_backbone.backbone.stages.0.attn.fc',
            #    'img_backbone.backbone.stages.1.attn.fc'
            #    'img_backbone.backbone.stages.2.attn.fc'
            #    'img_backbone.backbone.stages.3.attn.fc'] #['fc1', 'fc2']   # include all fc layers
    paddle_state_dict = {}
    for k in torch_state_dict:
        # print("k: ", k)
        # k = k.replace('module.', '')
        # if not (k.startswith('module.pts_bbox_head')):
        #     continue
        
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        # =========================================
        # modify for fc, transpose weight tensor
        flag = [i in k for i in fc_names]
        if any(flag) and k.endswith('weight'): # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print("ori shape: ", v.shape)
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        # -----------------------------------------
        # ===============================
        # modify for bn
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # -------------------------------


        #if k not in model_state_dict:
        if False:
            print(k)
        else:
            k = k.replace("module.", "")
            print("k: ", k)
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)

if __name__ == "__main__":
    torch2paddle()
