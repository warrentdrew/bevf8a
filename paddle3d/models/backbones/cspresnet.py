import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal
from paddle3d.models.layers import param_init
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.apis import manager

trunc_normal_ = TruncatedNormal(std=.02)

class Swish(nn.Layer):
    def forward(self, x):
        return x * F.sigmoid(x)


def get_activation(name="silu"):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == "silu":
            module = nn.SiLU()
        elif name == "relu":
            module = nn.ReLU()
        elif name == "lrelu":
            module = nn.LeakyReLU(0.1)
        elif name == 'swish':
            module = Swish()
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid()
        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module
    elif isinstance(name, nn.Layer):
        return name
    else:
        raise AttributeError("Unsupported act type: {}".format(name))


class ConvBNLayer(nn.Layer):

    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False
        )
        self.bn = nn.BatchNorm2D(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class EffectiveSELayer(nn.Layer):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2D(channels, channels, kernel_size=1, padding=0)
        self.act = get_activation(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class CSPResStage(nn.Layer):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 attn='eca',
                 use_alpha=False):
        super(CSPResStage, self).__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2, ch_mid // 2, act=act, shortcut=True, use_alpha=use_alpha)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = paddle.concat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class RepVggBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, act='relu', alpha=False, deploy=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.deploy = deploy
        if self.deploy == False:
            if alpha:
                x = paddle.zeros([1], dtype="float32")
                self.alpha = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
                self.alpha.stop_gradient = False
            else:
                self.alpha = None
            self.conv1 = ConvBNLayer(
                ch_in, ch_out, 3, stride=1, padding=1, act=None)
            self.conv2 = ConvBNLayer(
                ch_in, ch_out, 1, stride=1, padding=0, act=None)
        else:
            self.conv = nn.Conv2D(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1
            )
        self.act = get_activation(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        if self.deploy:
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def switch_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2D(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1
            )
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.set_value(kernel)
        self.conv.bias.set_value(bias)
        for para in self.parameters():
            para.detach()
        self.__delattr__(self.conv1)
        self.__delattr__(self.conv2)
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape([-1, 1, 1, 1])
        return kernel * t, beta - running_mean * gamma / std


class BasicBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, act='relu', shortcut=True, use_alpha=False):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act, alpha=use_alpha)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class CSPResNet(nn.Layer):

    def __init__(self,
                 layers=[3, 6, 6, 3],
                 channels=[64, 128, 256, 512, 1024],
                 act='swish',
                 return_idx=[0, 1, 2, 3],
                 use_large_stem=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 use_alpha=False):
        super().__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]

        if use_large_stem:
            from collections import OrderedDict
            self.stem = nn.Sequential(
                ('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act)),
                ('conv2', ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act)),
                ('conv3', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act)))
        else:
            from collections import OrderedDict
            self.stem = nn.Sequential(
                ('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act)),
                ('conv2', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act)))
        n = len(channels) - 1
        self.stages = nn.Sequential(
            *[CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act, use_alpha=use_alpha) for i in range(n)]
        )
        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

    def forward(self, inputs):
        x = inputs
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs


class SPP(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 act='swish',
                 ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2D(
                kernel_size=size,
                stride=1,
                padding=size // 2,
                ceil_mode=False)
            self.add_sublayer('pool{}'.format(i),
                            pool
                            )
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = paddle.concat(outs, axis=1)

        y = self.conv(y)
        return y


class CSPStage(nn.Layer):
    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', spp=False):
        super(CSPStage, self).__init__()

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock':
                self.convs.add_sublayer(str(i), BasicBlock(next_ch_in, ch_mid, act=act, shortcut=False))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_sublayer(
                    'spp',
                    SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act)
                )
            next_ch_in = ch_mid
        # self.convs = nn.Sequential(*convs)
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = paddle.concat([y1, y2], axis=1)
        y = self.conv3(y)
        return y

#   out_channels: [768, 384, 192]
#   stage_num: 1
#   block_num: 3
#   act: 'swish'
#   spp: true

class CustomCSPPAN(nn.Layer):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[1024, 512, 256],
                 norm_type='bn',
                 act='leaky',
                 stage_fn='CSPStage',
                 block_fn='BasicBlock',
                 stage_num=1,
                 block_num=3,
                 spp=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 ):
        super().__init__()
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act = get_activation(act) if act is None or isinstance(act,
                                                               (str, dict)) else act
        self.num_blocks = len(in_channels)
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        self.fpn_stages = nn.LayerList()
        self.fpn_routes = nn.LayerList()


        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                if stage_fn == 'CSPStage':
                    stage.add_sublayer(
                        str(j),
                        CSPStage(block_fn,
                                 ch_in if j == 0 else ch_out,
                                 ch_out,
                                 block_num,
                                 act=act,
                                 spp=(spp and i == 0))
                    )
                else:
                    raise NotImplementedError

            self.fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                self.fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act))
            ch_pre = ch_out

        # pan_stages = []
        # pan_routes = []
        # for i in reversed(range(self.num_blocks - 1)):
        #     pan_routes.append(
        #         ConvBNLayer(
        #             ch_in=out_channels[i + 1],
        #             ch_out=out_channels[i + 1],
        #             filter_size=3,
        #             stride=2,
        #             padding=1,
        #             act=act))

        #     ch_in = out_channels[i] + out_channels[i + 1]
        #     ch_out = out_channels[i]
        #     stage = nn.Sequential()
        #     for j in range(stage_num):
        #         stage.add_sublayer(
        #             str(j),
        #             eval(stage_fn)(block_fn,
        #                            ch_in if j == 0 else ch_out,
        #                            ch_out,
        #                            block_num,
        #                            act=act,
        #                            spp=False))

        #     pan_stages.append(stage)

        # self.pan_stages = nn.Sequential(*pan_stages[::-1])
        # self.pan_routes = nn.Sequential(*pan_routes[::-1])

    def forward(self, blocks):
        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = paddle.concat([route, block], axis=1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2.)
        return fpn_feats[-1]

        # pan_feats = [fpn_feats[-1], ]
        # route = fpn_feats[-1]
        # for i in reversed(range(self.num_blocks - 1)):
        #     block = fpn_feats[i]
        #     route = self.pan_routes[i](route)
        #     block = paddle.concat([route, block], axis=1)
        #     route = self.pan_stages[i](block)
        #     pan_feats.append(route)

        # return pan_feats[::-1]


@manager.BACKBONES.add_component
class CSPResNetFPN(nn.Layer):
    def __init__(self,
                layers=[3, 6, 6, 3],
                channels=[64, 128, 256, 512, 1024],
                act='swish', 
                return_idx=[0, 1, 2, 3],
                width_mult=0.50,
                depth_mult=0.33,
                use_large_stem=True,
                use_alpha=True,
                out_channels=[1024, 512, 256, 128],
                stage_num=1,
                block_num=3,
                spp=True,):
        super(CSPResNetFPN, self).__init__()
        self.backbone = CSPResNet(
            layers=layers,
            channels=channels,
            act=act, 
            return_idx=return_idx,
            width_mult=width_mult,
            depth_mult=depth_mult,
            use_large_stem=use_large_stem,
            use_alpha=use_alpha,
            )
        self.neck = CustomCSPPAN(
            in_channels=[channels[i+1] for i in return_idx],
            out_channels=out_channels,
            stage_num=stage_num,
            block_num=block_num,
            act=act,
            spp=spp,
            width_mult=width_mult,
            depth_mult=depth_mult,
        )
        self.init_weights()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    param_init.constant_init(m.bias, value=0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
                param_init.constant_init(m.bias, value=0)
                param_init.constant_init(m.weight, value=1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_pretrained_model(self, pretrained, verbose=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return (x,)
