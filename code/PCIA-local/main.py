"""
External Version 1.3
Internal Version 5.3
2021/04/30 by Joshua Wem & BWH & WZH from HEU & XiaoLong
"""
import sys
import time
import torch

from lib.para import Parameters
from lib.opt import opts
from lib.run import Run, Logger

time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
sys.stdout = Logger('./log/inventory/' + time_stamp + '.log')

if __name__ == '__main__':
    macAddress = 'd8bbc10b3b21'
    p = Parameters(macAddress=macAddress).init()
    p.camera_path = 'rtsp://admin:mcdz1234@192.168.1.165:554/Streaming/Channels/101'
    opt = opts().init()
    opt.load_model = './weights/pcia_v7_b'
    Run(parameters=p, opt=opt).run()
