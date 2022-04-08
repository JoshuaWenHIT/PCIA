from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--task', default='ctdet',
                                 help='ctdet | ddd | multi_pose | exdet')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--debug', type=int, default=0,
                                 help='level of visualization.'
                                      '1: only show the final detection results'
                                      '2: show the network output features'
                                      '3: use matplot to display'  # useful when lunching training with ipython notebook
                                      '4: save all visualizations to disk')
        self.parser.add_argument('--demo',
                                 default='./video/test2.mp4',
                                 help='path to image/ image folders/ video. '
                                      'or "webcam"')
        self.parser.add_argument('--stream',
                                 default='',
                                 help='rtmp | rtsp | udp | 0 "')
        self.parser.add_argument('--load_model', default='./weights/ap_b_v6',
                                 help='path to pretrained model')

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=12,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', default=True, action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.4,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='black',
                                 choices=['white', 'black'])

        # model
        self.parser.add_argument('--arch', default='resdcn_18',
                                 help='model architecture. Currently tested'
                                      'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                      'dlav0_34 | dla_34 | hourglass')
        self.parser.add_argument('--head_conv', type=int, default=64,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '64 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # test
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--test_scales', type=str, default='1',
                                 help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=100,
                                 help='max number of output objects.')
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')
        self.parser.add_argument('--overlap', type=float, default=0.55)
        self.parser.add_argument('--direction', type=str, default='r2l', help='r2l or l2r')

        # task
        # ctdet
        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh', action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not regress local offset.')

    def parse(self, args=''):
        if args == '':
            # opt = self.parser.parse_args()
            opt = self.parser.parse_known_args()[0]
        else:
            # opt = self.parser.parse_args(args)
            opt = self.parser.parse_known_args()[0]

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]
        opt.fix_res = not opt.keep_res
        opt.reg_offset = not opt.not_reg_offset

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 64
        opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

        if opt.debug > 0:
            # opt.num_workers = 0
            opt.num_workers = 8
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'ctdet':
            opt.heads = {'hm': opt.num_classes,
                         'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
        else:
            assert 0, 'task not defined!'
        return opt

    def init(self, args=''):
        default_dataset_info = {
            'ctdet': {'default_resolution': [512, 512], 'num_classes': 2,
                      # 'mean': [0.38211868197971827, 0.3909553800072519, 0.43377550115921054],
                      'mean': [0.38211868197971825, 0.3909553800072519, 0.43377550115921054],
                      # 'std': [0.19241244124910198, 0.18910680579742356, 0.18762484722890635],
                      'std': [0.18761946051581851, 0.1958170047166515, 0.21482773681590914],
                      'dataset': 'pig'},
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
