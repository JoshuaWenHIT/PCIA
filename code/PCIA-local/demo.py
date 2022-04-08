"""
External Version 2.1
Internal Version 4.1
2021/05/25 by Joshua Wem from HEU & XiaoLong
"""
import torch
from lib.opt import opts
from lib.para import Parameters
from lib.demo import demo


if __name__ == '__main__':
    macAddress = ''
    p = Parameters(macAddress=macAddress).init()
    opt = opts().init()
    opt.load_model = './weights/pcia_v7_b'
    opt.demo = './video/test2.mp4'
    demo(opt, p)
