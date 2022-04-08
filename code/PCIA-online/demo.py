"""
External Version 2.1
Internal Version 4.1
2021/05/25 by Joshua Wem from HEU & XiaoLong
"""
from lib.opt import opts
from lib.para import Parameters
from lib.demo import demo


if __name__ == '__main__':
    macAddress = ''
    p = Parameters(macAddress=macAddress).init()
    opt = opts().init()
    opt.load_model = './weights/ap_b_v6'
    opt.vis_thresh = 0.5
    opt.direction = 'l2r'
    opt.demo = './video/test2.mp4'
    # opt.stream = 'rtmp://127.0.0.1:1935/myapp/test'
    demo(opt, p)
