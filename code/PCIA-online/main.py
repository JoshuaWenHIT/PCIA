"""
External Version 1.3
Internal Version 5.3
2021/04/30 by Joshua Wem & BWH & WZH from HEU & XiaoLong
"""
import sys
import time

from lib.para import Parameters
from lib.opt import opts
from lib.run import Run, Logger

time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
sys.stdout = Logger('./log/inventory/' + time_stamp + '.log')


if __name__ == '__main__':
    macAddress = ''
    p = Parameters(macAddress=macAddress).init()
    opt = opts().init()
    opt.direction = 'r2l'
    p.camera_path = 'rtsp://admin:guocheng660@192.168.1.64:554/Streaming/Channels/101'
    Run(parameters=p, opt=opt).run()
