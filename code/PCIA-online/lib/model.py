"""
External Version 1.0
Internal Version 2.0
2021/04/30 by Joshua Wem from HEU & XiaoLong
"""
import os
import tempfile

import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable
import time
import platform
import lib._ext as _backend

import datetime
import socket
import struct
import binascii
from Crypto.Cipher import AES


# ENCODE
def get_time(flag=0):
    datetimenow = datetime.datetime.now()
    date = datetimenow.date().isoformat()
    time = datetimenow.time().strftime('%H-%M-%S-%f')

    if flag == 0:
        return date + "-" + time
    if flag == 1:
        return date
    if flag == 2:
        return time


class Get_License(object):
    def __init__(self):
        super(Get_License, self).__init__()

        self.seperateKey = "JoshuaWen66"
        self.aesKey = "jojojojo12345678"
        self.aesIv = "jo19950315dragon"
        self.aesMode = AES.MODE_CBC

    def getHwAddr(self, ifname):
        """
        获取主机物理地址
        """
        if platform.architecture()[1] == 'ELF' or platform.architecture()[1] == '':
            import fcntl
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            info = fcntl.ioctl(s.fileno(), 0x8927, struct.pack('256s', bytes(ifname[:15], 'utf-8')))
            return ''.join(['%02x' % char for char in info[18:24]])
        elif platform.architecture()[1] == 'WindowsPE':
            import uuid
            return hex(uuid.getnode())[2:]

    def decrypt(self, text):
        try:
            cryptor = AES.new(self.aesKey.encode('utf-8'), self.aesMode, self.aesIv.encode('utf-8'))
            plain_text = cryptor.decrypt(binascii.unhexlify(text))
            return plain_text
        except:
            return ""

    def getLicenseInfo(self, filePath=None):
        if filePath is None:
            filePath = "./license.lic"

        if not os.path.isfile(filePath):
            print("请将 license.lic 文件放在当前路径下")
            os._exit(0)
            return False, 'Invalid'

        encryptText = ""
        with open(filePath, "rb") as licFile:
            encryptText = licFile.read()
            licFile.close()
        try:
            hostInfo = self.getHwAddr('eno1')

        except IOError:
            hostInfo = self.getHwAddr('wlp4s0')

        decryptText = self.decrypt(encryptText)
        pos = decryptText.decode().find(self.seperateKey)
        if -1 == pos:
            return False, "Invalid"
        licHostInfo = self.decrypt((decryptText[0:pos]).decode())
        licHostInfo = licHostInfo.decode().rstrip('6')
        true_licHostInfo = licHostInfo.split('-')[0]
        licenseStr = decryptText[pos + len(self.seperateKey):]

        date = get_time(1)
        lic_time = licHostInfo.split('-')
        true_time = date.split('-')
        if int(true_time[1]) < int(lic_time[1]):
            if true_licHostInfo == hostInfo:
                return True, licenseStr
            else:
                return False, 'Invalid'
        elif int(true_time[1]) == int(lic_time[1]):
            if int(true_time[2]) < int(lic_time[2]):
                return True, licenseStr
            else:
                return False, 'Invalid'
        else:
            return False, 'Invalid'


def lic_match():
    License = Get_License()
    condition, LicInfo = License.getLicenseInfo()
    if condition == True and LicInfo == b'Valid':
        print("已授权！")
        return True
    else:
        print('未权授！')
        return False


# FUNCTION
def model_builder(opt):
    assert lic_match()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug,
                    1)
    Detector = detector_factory[
        opt.task]
    detector = Detector(opt)
    detector.pause = False
    return detector


def detect_and_count(opt, img, detector, counter, dets_pre, width, height):
    ret = detector.run(img)
    ret_torch = torch.from_numpy(ret['results'][1])
    dets = ret_torch.new()
    x_line1 = (width / 2, 0)
    y_line1 = (width / 2, height)
    for bbox in ret['results'][1]:
        if bbox[4] > opt.vis_thresh:
            bbox_torch = torch.from_numpy(bbox[:4])
            dets = torch.cat((dets, bbox_torch.unsqueeze(0)), 0)
            draw_bbox(img, bbox[:4], 0, bbox[4])

    for bbox in ret['results'][2]:
        if bbox[4] > opt.vis_thresh:
            draw_bbox(img, bbox[:4], 1, bbox[4])

    if not dets_pre.numel():
        boxes_pair_idx = []
    else:
        boxes_pair_idx = cal_IoU_between_two_boxes(dets_pre, dets, overlap=opt.overlap)

    for pair in boxes_pair_idx:
        dets_pre_idx = pair[0]
        dets_idx = pair[1]

        x1_pre = dets_pre[dets_pre_idx][0]
        y1_pre = dets_pre[dets_pre_idx][1]
        x2_pre = dets_pre[dets_pre_idx][2]
        y2_pre = dets_pre[dets_pre_idx][3]

        x1 = dets[dets_idx][0]
        y1 = dets[dets_idx][1]
        x2 = dets[dets_idx][2]
        y2 = dets[dets_idx][3]

        x_core_pre = (x1_pre + x2_pre) / 2
        y_core_pre = (y1_pre + y2_pre) / 2
        x_core = (x1 + x2) / 2
        y_core = (y1 + y2) / 2
        core_pre = (x_core_pre, y_core_pre)
        core = (x_core, y_core)

        if __intersect(core_pre, core, x_line1, y_line1):
            if opt.direction == 'r2l':
                if x_core > x_core_pre:
                    counter -= 1  # right to left
                elif x_core < x_core_pre:
                    counter += 1  # right to left
            if opt.direction == 'l2r':
                if x_core > x_core_pre:
                    counter += 1  # right to left
                elif x_core < x_core_pre:
                    counter -= 1  # right to left

    cv2.putText(img, 'Total: {:d}'.format(counter), (int(width / 2), int(width / 2)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 0), 2, cv2.LINE_AA)

    # cv2.line(img, x_line1, y_line1, (255, 0, 0), 2, cv2.LINE_AA)

    return img, counter, dets, ret


# MODEL
def create_model(arch, heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    key = b'JoshuaWen6'
    w_input = open(model_path, 'rb')
    data_output = []
    key_len = len(key)
    while True:
        data = w_input.read(key_len)
        if not data:
            break
        for i in range(len(data)):
            data_output.append(data[i] ^ key[i])
    w_input.close()
    w_output = tempfile.NamedTemporaryFile(delete=False)
    w_output.write(bytes(data_output))
    checkpoint = torch.load(w_output.name, map_location=lambda storage, loc: storage)
    # print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    print('加载模型 {}'.format(model_path))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


# NETWORK
BN_MOMENTUM = 0.1
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(64, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes,
                     kernel_size=(3, 3), stride=1,
                     padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1,
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

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
        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        return [ret]

    def init_weights(self, num_layers):
        if 1:
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            # print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
            # print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


class _DCNv2(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(input, weight, bias,
                                         offset, mask,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            _backend.dcn_v2_backward(input, weight,
                                     bias,
                                     offset, mask,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.deformable_groups)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, \
               None, None, None, None,


dcn_v2_conv = _DCNv2.apply


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
               offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
               mask.shape[1]
        return dcn_v2_conv(input, offset, mask,
                           self.weight,
                           self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask,
                           self.weight, self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)


class _DCNv2Pooling(Function):
    @staticmethod
    def forward(ctx, input, rois, offset,
                spatial_scale,
                pooled_size,
                output_dim,
                no_trans,
                group_size=1,
                part_size=None,
                sample_per_part=4,
                trans_std=.0):
        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        output, output_count = \
            _backend.dcn_v2_psroi_pooling_forward(input, rois, offset,
                                                  ctx.no_trans, ctx.spatial_scale,
                                                  ctx.output_dim, ctx.group_size,
                                                  ctx.pooled_size, ctx.part_size,
                                                  ctx.sample_per_part, ctx.trans_std)
        ctx.save_for_backward(input, rois, offset, output_count)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = \
            _backend.dcn_v2_psroi_pooling_backward(grad_output,
                                                   input,
                                                   rois,
                                                   offset,
                                                   output_count,
                                                   ctx.no_trans,
                                                   ctx.spatial_scale,
                                                   ctx.output_dim,
                                                   ctx.group_size,
                                                   ctx.pooled_size,
                                                   ctx.part_size,
                                                   ctx.sample_per_part,
                                                   ctx.trans_std)

        return grad_input, None, grad_offset, \
               None, None, None, None, None, None, None, None


dcn_v2_pooling = _DCNv2Pooling.apply


class DCNv2Pooling(nn.Module):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            offset = input.new()
        return dcn_v2_pooling(input, rois, offset,
                              self.spatial_scale,
                              self.pooled_size,
                              self.output_dim,
                              self.no_trans,
                              self.group_size,
                              self.part_size,
                              self.sample_per_part,
                              self.trans_std)


class DCNPooling(DCNv2Pooling):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_dim=1024):
        super(DCNPooling, self).__init__(spatial_scale,
                                         pooled_size,
                                         output_dim,
                                         no_trans,
                                         group_size,
                                         part_size,
                                         sample_per_part,
                                         trans_std)

        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.offset_mask_fc = nn.Sequential(
                nn.Linear(self.pooled_size * self.pooled_size *
                          self.output_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_dim, self.pooled_size *
                          self.pooled_size * 3)
            )
            self.offset_mask_fc[4].weight.data.zero_()
            self.offset_mask_fc[4].bias.data.zero_()

    def forward(self, input, rois):
        offset = input.new()

        if not self.no_trans:
            # do roi_align first
            n = rois.shape[0]
            roi = dcn_v2_pooling(input, rois, offset,
                                 self.spatial_scale,
                                 self.pooled_size,
                                 self.output_dim,
                                 True,  # no trans
                                 self.group_size,
                                 self.part_size,
                                 self.sample_per_part,
                                 self.trans_std)

            # build mask and offset
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(
                n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)

            # do pooling with offset and mask
            return dcn_v2_pooling(input, rois, offset,
                                  self.spatial_scale,
                                  self.pooled_size,
                                  self.output_dim,
                                  self.no_trans,
                                  self.group_size,
                                  self.part_size,
                                  self.sample_per_part,
                                  self.trans_std) * mask
        # only roi_align
        return dcn_v2_pooling(input, rois, offset,
                              self.spatial_scale,
                              self.pooled_size,
                              self.output_dim,
                              self.no_trans,
                              self.group_size,
                              self.part_size,
                              self.sample_per_part,
                              self.trans_std)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_pose_net(num_layers, heads, head_conv=256):
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
    model.init_weights(num_layers)
    return model


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}
_model_factory = {
    'resdcn': get_pose_net,
}


# DETECTOR
class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('生成模型...')
        self.model = create_model(opt.arch, opt.heads,
                                  opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor,
                      np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                t1 = time.time()
                images, meta = self.pre_process(image, scale, meta)
                t2 = time.time()
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            t3 = time.time()
            images = images.to(self.opt.device)
            t4 = time.time()
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time
            t12 = '{:.6}'.format(t2-t1)
            t23 = '{:.6}'.format(t3-t2)
            t34 = '{:.6}'.format(t4-t3)
            print(t12,t23,t34,sep=' ',end='\n')

            output, dets, forward_time = self.process(images, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)

            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results)

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}


class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'],
            self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(
                np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
        # debugger.show_all_imgs(pause=self.pause)
        # debugger.save_all_imgs(path='./temp', genID=True)


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)],
                axis=1).tolist()
        ret.append(top_preds)
    return ret


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


detector_factory = {
    'ctdet': CtdetDetector,
}


# DEBUGGER
class Debugger(object):
    def __init__(self, ipynb=False, theme='black',
                 num_classes=-1, dataset=None, down_ratio=4):
        self.ipynb = ipynb
        if not self.ipynb:
            import matplotlib.pyplot as plt
            self.plt = plt
        self.imgs = {}
        self.theme = theme
        colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        if self.theme == 'white':
            self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
            self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
        self.dim_scale = 1
        if dataset == 'coco_hp':
            self.names = ['p']
            self.num_class = 1
            self.num_joints = 17
            self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                          [3, 5], [4, 6], [5, 6],
                          [5, 7], [7, 9], [6, 8], [8, 10],
                          [5, 11], [6, 12], [11, 12],
                          [11, 13], [13, 15], [12, 14], [14, 16]]
            self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                       (255, 0, 0), (0, 0, 255), (255, 0, 255),
                       (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                       (255, 0, 0), (0, 0, 255), (255, 0, 255),
                       (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
            self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
                              (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                              (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                              (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                              (255, 0, 0), (0, 0, 255)]
        elif num_classes == 80 or dataset == 'coco':
            self.names = coco_class_name
        elif num_classes == 20 or dataset == 'pascal':
            self.names = pascal_class_name
        elif num_classes == 2 or dataset == 'pig':
            self.names = pig_class_name
        elif num_classes == 6 or dataset == 'ship':
            self.names = ship_class_name
        elif dataset == 'gta':
            self.names = gta_class_name
            self.focal_length = 935.3074360871937
            self.W = 1920
            self.H = 1080
            self.dim_scale = 3
        elif dataset == 'viper':
            self.names = gta_class_name
            self.focal_length = 1158
            self.W = 1920
            self.H = 1080
            self.dim_scale = 3
        elif num_classes == 3 or dataset == 'kitti':
            self.names = kitti_class_name
            self.focal_length = 721.5377
            self.W = 1242
            self.H = 375
        num_classes = len(self.names)
        self.down_ratio = down_ratio
        # for bird view
        self.world_size = 64
        self.out_size = 384
        # for video saving
        # self.video_saved_width = 1280
        # self.video_saved_height = 720
        # self.fps = 15
        # self.video_saved_dir = '/home/joshuawen/Projects/CenterNet/video/video_det_results'
        # self.opt = opts().init()
        # self.video_saved_name = self.opt.demo.split('/')[-1].split('.')[0] + '_res' + '.mp4'

    def add_img(self, img, img_id='default', revert_color=False):
        if revert_color:
            img = 255 - img
        self.imgs[img_id] = img.copy()

    def add_mask(self, mask, bg, imgId='default', trans=0.8):
        self.imgs[imgId] = (mask.reshape(
            mask.shape[0], mask.shape[1], 1) * 255 * trans + \
                            bg * (1 - trans)).astype(np.uint8)

    def show_img(self, pause=False, imgId='default'):
        cv2.imshow('{}'.format(imgId), self.imgs[imgId])
        if pause:
            cv2.waitKey()

    def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
        if self.theme == 'white':
            fore = 255 - fore
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = (back * (1. - trans) + fore * trans)
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

    '''
  # slow version
  def gen_colormap(self, img, output_res=None):
    # num_classes = len(self.colors)
    img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl = self.colors[i] if not (self.theme == 'white') \
           else 255 - self.colors[i]
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
    '''

    def gen_colormap(self, img, output_res=None):
        img = img.copy()
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(
            self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        if self.theme == 'white':
            colors = 255 - colors
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map

    '''
  # slow
  def gen_colormap_hp(self, img, output_res=None):
    # num_classes = len(self.colors)
    # img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl =  self.colors_hp[i] if not (self.theme == 'white') else \
        (255 - np.array(self.colors_hp[i]))
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
  '''

    def gen_colormap_hp(self, img, output_res=None):
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(
            self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        if self.theme == 'white':
            colors = 255 - colors
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map

    def add_rect(self, rect1, rect2, c, conf=1, img_id='default'):
        cv2.rectangle(
            self.imgs[img_id], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 2)
        if conf < 1:
            cv2.circle(self.imgs[img_id], (rect1[0], rect1[1]), int(10 * conf), c, 1)
            cv2.circle(self.imgs[img_id], (rect2[0], rect2[1]), int(10 * conf), c, 1)
            cv2.circle(self.imgs[img_id], (rect1[0], rect2[1]), int(10 * conf), c, 1)
            cv2.circle(self.imgs[img_id], (rect2[0], rect1[1]), int(10 * conf), c, 1)

    def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'):
        bbox = np.array(bbox, dtype=np.int32)
        # cat = (int(cat) + 1) % 80
        cat = int(cat)
        # print('cat', cat, self.names[cat])
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()
        txt = '{}{:.1f}'.format(self.names[cat], conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(
            self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
        if show_txt:
            cv2.rectangle(self.imgs[img_id],
                          (bbox[0], bbox[1] - cat_size[1] - 2),
                          (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2),
                        font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    def add_coco_hp(self, points, img_id='default'):
        points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
        for j in range(self.num_joints):
            cv2.circle(self.imgs[img_id],
                       (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
        for j, e in enumerate(self.edges):
            if points[e].min() > 0:
                cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
                         (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
                         lineType=cv2.LINE_AA)

    def add_points(self, points, img_id='default'):
        num_classes = len(points)
        # assert num_classes == len(self.colors)
        for i in range(num_classes):
            for j in range(len(points[i])):
                c = self.colors[i, 0, 0]
                cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
                                               points[i][j][1] * self.down_ratio),
                           5, (255, 255, 255), -1)
                cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
                                               points[i][j][1] * self.down_ratio),
                           3, (int(c[0]), int(c[1]), int(c[2])), -1)

    def show_all_imgs(self, pause=False, time=0):
        # for video saving
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # video_saved_path = os.path.join(self.video_saved_dir, self.video_saved_name)
        # video_saved = cv2.VideoWriter(video_saved_path, fourcc, self.fps,
        #                               (self.video_saved_width, self.video_saved_height))
        if not self.ipynb:
            for i, v in self.imgs.items():
                cv2.imshow('{}'.format(i), v)
                # video_saved.write(v)
            # video_saved.release()
            if cv2.waitKey(0 if pause else 1) == 27:
                # video_saved.release()
                import sys
                sys.exit(0)
        else:
            self.ax = None
            nImgs = len(self.imgs)
            fig = self.plt.figure(figsize=(nImgs * 10, 10))
            nCols = nImgs
            nRows = nImgs // nCols
            for i, (k, v) in enumerate(self.imgs.items()):
                fig.add_subplot(1, nImgs, i + 1)
                if len(v.shape) == 3:
                    self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                else:
                    self.plt.imshow(v)
            self.plt.show()

    def save_img(self, imgId='default', path='./cache/debug/'):
        cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])

    def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
        if genID:
            try:
                idx = int(np.loadtxt(path + '/id.txt'))
            except:
                idx = 0
            prefix = idx
            np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
        for i, v in self.imgs.items():
            cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

    def remove_side(self, img_id, img):
        if not (img_id in self.imgs):
            return
        ws = img.sum(axis=2).sum(axis=0)
        l = 0
        while ws[l] == 0 and l < len(ws):
            l += 1
        r = ws.shape[0] - 1
        while ws[r] == 0 and r > 0:
            r -= 1
        hs = img.sum(axis=2).sum(axis=1)
        t = 0
        while hs[t] == 0 and t < len(hs):
            t += 1
        b = hs.shape[0] - 1
        while hs[b] == 0 and b > 0:
            b -= 1
        self.imgs[img_id] = self.imgs[img_id][t:b + 1, l:r + 1].copy()

    def project_3d_to_bird(self, pt):
        pt[0] += self.world_size / 2
        pt[1] = self.world_size - pt[1]
        pt = pt * self.out_size / self.world_size
        return pt.astype(np.int32)

    def add_ct_detection(
            self, img, dets, show_box=False, show_txt=True,
            center_thresh=0.5, img_id='det'):
        # dets: max_preds x 5
        self.imgs[img_id] = img.copy()
        if type(dets) == type({}):
            for cat in dets:
                for i in range(len(dets[cat])):
                    if dets[cat][i, 2] > center_thresh:
                        cl = (self.colors[cat, 0, 0]).tolist()
                        ct = dets[cat][i, :2].astype(np.int32)
                        if show_box:
                            w, h = dets[cat][i, -2], dets[cat][i, -1]
                            x, y = dets[cat][i, 0], dets[cat][i, 1]
                            bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                                            dtype=np.float32)
                            self.add_coco_bbox(
                                bbox, cat - 1, dets[cat][i, 2],
                                show_txt=show_txt, img_id=img_id)
        else:
            for i in range(len(dets)):
                if dets[i, 2] > center_thresh:
                    # print('dets', dets[i])
                    cat = int(dets[i, -1])
                    cl = (self.colors[cat, 0, 0] if self.theme == 'black' else \
                              255 - self.colors[cat, 0, 0]).tolist()
                    ct = dets[i, :2].astype(np.int32) * self.down_ratio
                    cv2.circle(self.imgs[img_id], (ct[0], ct[1]), 3, cl, -1)
                    if show_box:
                        w, h = dets[i, -3] * self.down_ratio, dets[i, -2] * self.down_ratio
                        x, y = dets[i, 0] * self.down_ratio, dets[i, 1] * self.down_ratio
                        bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                                        dtype=np.float32)
                        self.add_coco_bbox(bbox, dets[i, -1], dets[i, 2], img_id=img_id)

    def add_3d_detection(
            self, image_or_path, dets, calib, show_txt=False,
            center_thresh=0.5, img_id='det'):
        if isinstance(image_or_path, np.ndarray):
            self.imgs[img_id] = image_or_path
        else:
            self.imgs[img_id] = cv2.imread(image_or_path)
        for cat in dets:
            for i in range(len(dets[cat])):
                cl = (self.colors[cat - 1, 0, 0]).tolist()
                if dets[cat][i, -1] > center_thresh:
                    dim = dets[cat][i, 5:8]
                    loc = dets[cat][i, 8:11]
                    rot_y = dets[cat][i, 11]
                    # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
                    # dim = dim / self.dim_scale
                    if loc[2] > 1:
                        box_3d = compute_box_3d(dim, loc, rot_y)
                        box_2d = project_to_image(box_3d, calib)
                        self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)

    def compose_vis_add(
            self, img_path, dets, calib,
            center_thresh, pred, bev, img_id='out'):
        self.imgs[img_id] = cv2.imread(img_path)
        # h, w = self.imgs[img_id].shape[:2]
        # pred = cv2.resize(pred, (h, w))
        h, w = pred.shape[:2]
        hs, ws = self.imgs[img_id].shape[0] / h, self.imgs[img_id].shape[1] / w
        self.imgs[img_id] = cv2.resize(self.imgs[img_id], (w, h))
        self.add_blend_img(self.imgs[img_id], pred, img_id)
        for cat in dets:
            for i in range(len(dets[cat])):
                cl = (self.colors[cat - 1, 0, 0]).tolist()
                if dets[cat][i, -1] > center_thresh:
                    dim = dets[cat][i, 5:8]
                    loc = dets[cat][i, 8:11]
                    rot_y = dets[cat][i, 11]
                    # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
                    # dim = dim / self.dim_scale
                    if loc[2] > 1:
                        box_3d = compute_box_3d(dim, loc, rot_y)
                        box_2d = project_to_image(box_3d, calib)
                        box_2d[:, 0] /= hs
                        box_2d[:, 1] /= ws
                        self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)
        self.imgs[img_id] = np.concatenate(
            [self.imgs[img_id], self.imgs[bev]], axis=1)

    def add_2d_detection(
            self, img, dets, show_box=False, show_txt=True,
            center_thresh=0.5, img_id='det'):
        self.imgs[img_id] = img
        for cat in dets:
            for i in range(len(dets[cat])):
                cl = (self.colors[cat - 1, 0, 0]).tolist()
                if dets[cat][i, -1] > center_thresh:
                    bbox = dets[cat][i, 1:5]
                    self.add_coco_bbox(
                        bbox, cat - 1, dets[cat][i, -1],
                        show_txt=show_txt, img_id=img_id)

    def add_bird_view(self, dets, center_thresh=0.3, img_id='bird'):
        bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
        for cat in dets:
            cl = (self.colors[cat - 1, 0, 0]).tolist()
            lc = (250, 152, 12)
            for i in range(len(dets[cat])):
                if dets[cat][i, -1] > center_thresh:
                    dim = dets[cat][i, 5:8]
                    loc = dets[cat][i, 8:11]
                    rot_y = dets[cat][i, 11]
                    rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
                    for k in range(4):
                        rect[k] = self.project_3d_to_bird(rect[k])
                        # cv2.circle(bird_view, (rect[k][0], rect[k][1]), 2, lc, -1)
                    cv2.polylines(
                        bird_view, [rect.reshape(-1, 1, 2).astype(np.int32)],
                        True, lc, 2, lineType=cv2.LINE_AA)
                    for e in [[0, 1]]:
                        t = 4 if e == [0, 1] else 1
                        cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                                 (rect[e[1]][0], rect[e[1]][1]), lc, t,
                                 lineType=cv2.LINE_AA)
        self.imgs[img_id] = bird_view

    def add_bird_views(self, dets_dt, dets_gt, center_thresh=0.3, img_id='bird'):
        alpha = 0.5
        bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
        for ii, (dets, lc, cc) in enumerate(
                [(dets_gt, (12, 49, 250), (0, 0, 255)),
                 (dets_dt, (250, 152, 12), (255, 0, 0))]):
            # cc = np.array(lc, dtype=np.uint8).reshape(1, 1, 3)
            for cat in dets:
                cl = (self.colors[cat - 1, 0, 0]).tolist()
                for i in range(len(dets[cat])):
                    if dets[cat][i, -1] > center_thresh:
                        dim = dets[cat][i, 5:8]
                        loc = dets[cat][i, 8:11]
                        rot_y = dets[cat][i, 11]
                        rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
                        for k in range(4):
                            rect[k] = self.project_3d_to_bird(rect[k])
                        if ii == 0:
                            cv2.fillPoly(
                                bird_view, [rect.reshape(-1, 1, 2).astype(np.int32)],
                                lc, lineType=cv2.LINE_AA)
                        else:
                            cv2.polylines(
                                bird_view, [rect.reshape(-1, 1, 2).astype(np.int32)],
                                True, lc, 2, lineType=cv2.LINE_AA)
                        # for e in [[0, 1], [1, 2], [2, 3], [3, 0]]:
                        for e in [[0, 1]]:
                            t = 4 if e == [0, 1] else 1
                            cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                                     (rect[e[1]][0], rect[e[1]][1]), lc, t,
                                     lineType=cv2.LINE_AA)
        self.imgs[img_id] = bird_view


def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)


def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homo = np.concatenate(
        [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    # import pdb; pdb.set_trace()
    return pts_2d


def draw_box_3d(image, corners, c=(0, 0, 255)):
    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), c, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                     (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                     (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
    return image


kitti_class_name = [
    'p', 'v', 'b'
]

gta_class_name = [
    'p', 'v'
]

pascal_class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                     "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                     "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

coco_class_name = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

pig_class_name = ['pig', 'person']

ship_class_name = ['liner', 'container ship', 'bulk carrier', 'island reef', 'sailboat', 'other ship']

color_list = np.array(
    [
        1.000, 1.000, 1.000,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255


# TOOLS
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def cal_IoU_between_two_boxes(boxes1, boxes2, overlap=0.8):
    idx_list = []
    N1 = boxes1.size(0)
    N2 = boxes2.size(0)

    for i in range(N1):
        for j in range(N2):
            iou = jaccard(boxes1[i].unsqueeze(0), boxes2[j].unsqueeze(0))
            if iou > overlap:
                idx_list.append((i, j))
    return idx_list


def __intersect(point_aa, point_bb,
                point_cc, point_dd):
    # this fuction will judge whether two line-segment is intersect
    # from https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    s10_x = point_bb[0] - point_aa[0]
    s10_y = point_bb[1] - point_aa[1]
    s32_x = point_dd[0] - point_cc[0]
    s32_y = point_dd[1] - point_cc[1]

    denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0:
        return False

    denomPositive = denom > 0

    s02_x = point_aa[0] - point_cc[0]
    s02_y = point_aa[1] - point_cc[1]
    s_numer = s10_x * s02_y - s10_y * s02_x
    if (s_numer < 0) == denomPositive:
        return False
    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == denomPositive:
        return False

    if ((s_numer > denom) == denomPositive) or ((t_numer > denom) == denomPositive):
        return False

    return True


def draw_bbox(img, bbox, cat, conf=1, show_txt=True):
    color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
    colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)

    names = ['pig', 'person']

    bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    c = colors[cat][0][0].tolist()
    txt = '{}{:.1f}'.format(names[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(
        img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)

    # For ablation experiments
    # cv2.circle(img, (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)), 4, (0, 0, 255), -1)

    if show_txt:
        cv2.rectangle(img,
                      (bbox[0], bbox[1] - cat_size[1] - 2),
                      (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
        cv2.putText(img, txt, (bbox[0], bbox[1] - 2),
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
