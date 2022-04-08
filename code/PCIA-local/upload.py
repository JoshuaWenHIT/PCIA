"""
External Version 1.0
Internal Version 1.0
2021/05/19 by Joshua Wem from HEU & FocusLoong & WHW from FocusLoong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
from lib.run import Logger

from lib.para import Parameters
from lib.upload import upload_v2

time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
sys.stdout = Logger('./log/upload/' + time_stamp + '.log')

if __name__ == '__main__':
    macAddress = ''
    p = Parameters(macAddress=macAddress).init()
    upload_results = upload_v2(p.appKey, p.secretKey, p)
