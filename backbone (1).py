import os

import torch
import torch.nn as nn
import torchvision
from .ada_agg import AdaAggLayer

experts = 5

__all__ = ['ResNet', 'build_model']


def conv7x7(in_planes, out_planes, padding=3, stride=1, groups=1, dilation=1, experts=5, align=False, lite=False):
    return AdaAggLayer(
        in_planes, out_planes, kernel_size=7, stride=stride, 
        padding=padding, groups=groups, bias=False, dilation=dilation,
        experts=experts, align=align, lite=lite
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, experts=5, align=False, lite=False):
    return AdaAggLayer(
        in_planes, out_planes, kernel_size=3, stride=stride, 
        padding=dilation, groups=groups, bias=False, dilation=dilation,
        experts=experts, align=align, lite=lite
    )


def conv1x1(in_planes, out_planes, stride=1, experts=5, align=False, lite=False):
    return AdaAggLayer(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False, 
        experts=experts, align=align, lite=lite
    )

# bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, experts=5, align=False, lite=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, experts=experts, align=align, lite=lite)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, experts=experts, align=align, lite=lite)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, experts=experts, align=align, lite=lite)
        self.bn3 = norm_layer(planes * self.expansion) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, experts=5, align=True, lite=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.experts = experts
        self.align = align
        self.lite = lite

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv7x7(3, self.inplanes, stride=2, padding=3, experts=experts, align=align, lite=lite)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 
                                       experts=experts, align=align, lite=lite)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], 
                                       experts=experts, align=align, lite=lite)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], 
                                       experts=experts, align=align, lite=lite)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], 
                                       experts=experts, align=align, lite=lite)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, experts=5, align=False, lite=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            experts=experts, align=align, lite=lite))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                experts=experts, align=align, lite=lite))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        # x = self.fc(x)

        return x


def load_pretrained_pkl(name, model_path='pretrained_models'):
    pretrained_model_path = {
        'imageNet': 'resnet50-0676ba61.pth',
        'moco': 'moco_v1_200ep_pretrain.pth.tar',
        'maskrcnn': 'maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        'deeplab': 'deeplabv3_resnet50_coco-cd0a2569.pth',
        'keyPoint': 'keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
    }
    print(pretrained_model_path[name])
    pkl = torch.load(os.path.join(model_path, pretrained_model_path[name]))
    state_dict = {}

    if name == 'imageNet':
        for k, v in pkl.items():
            if k.startswith("fc."):
                continue
            state_dict[k] = v
    elif name == 'moco':
        pkl = pkl['state_dict']
        state_dict = {}
        for k, v in pkl.items():
            if not k.startswith("module.encoder_q."):
                continue
            k = k.replace("module.encoder_q.", "")
            if k.startswith("fc."):
                continue
            state_dict[k] = v
    elif name == 'maskrcnn':
        for k, v in pkl.items():
            if not k.startswith("backbone.body."):
                continue
            k = k.replace("backbone.body.", "")
            state_dict[k] = v
    elif name == 'deeplab':
        for k, v in pkl.items():
            if not k.startswith("backbone."):
                continue
            k = k.replace("backbone.", "")
            state_dict[k] = v
    elif name == 'keyPoint':
        for k, v in pkl.items():
            if not k.startswith("backbone.body."):
                continue
            k = k.replace("backbone.body.", "")
            state_dict[k] = v

    return state_dict

def build_model(model_dict=None, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)

    if model_dict is None:
        state_dict_imageNet = load_pretrained_pkl('imageNet')
        state_dict_moco = load_pretrained_pkl('moco')
        state_dict_maskrcnn = load_pretrained_pkl('maskrcnn')
        state_dict_deeplab = load_pretrained_pkl('deeplab')
        state_dict_keyPoint = load_pretrained_pkl('keyPoint')
    else:
        state_dict_imageNet = model_dict['imageNet']
        state_dict_moco = model_dict['moco']
        state_dict_maskrcnn = model_dict['maskrcnn']
        state_dict_deeplab = model_dict['deeplab']
        state_dict_keyPoint = model_dict['keyPoint']

    state_dict_new = model.state_dict()

    for name in state_dict_imageNet:
        w1 = state_dict_imageNet[name]
        w2 = state_dict_moco[name]
        w3 = state_dict_maskrcnn[name]
        w4 = state_dict_deeplab[name]
        w5 = state_dict_keyPoint[name]

        wt = state_dict_new[name]

        if wt.size() == w1.size():
            # print('Same size:    ', name)
            state_dict_new[name] = (w1 + w2 + w3 + w4 + w5) / 5.0
        else:
            # print('Different size:    ', name)
            state_dict_new[name] = torch.cat((w1.unsqueeze(0), w2.unsqueeze(0), w3.unsqueeze(0),
                                                w4.unsqueeze(0), w5.unsqueeze(0)), dim=0)


        if 'mean' in name:
            state_dict_new[name] = w1 * 0.0
        elif 'var' in name:
            state_dict_new[name] = w1 * 0.0 + 1.0

    model.load_state_dict(state_dict_new)

    return model
