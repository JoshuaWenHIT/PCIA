from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import uuid
import hmac
import base64
from hashlib import sha1


class Parameters(object):
    def __init__(self, macAddress, rtmpIP = 'rtmp://114.215.147.82:1935/live/'):
        self.parser = argparse.ArgumentParser()

        # MQTT parameters
        self.parser.add_argument('--host', type=str, default='mqtt-cn-oew1w1cq50a.mqtt.aliyuncs.com')
        self.parser.add_argument('--port', type=int, default=1883)
        self.parser.add_argument('--keepalive', type=int, default=60)
        self.parser.add_argument('--instanceId', type=str, default='mqtt-cn-oew1w1cq50a')
        self.parser.add_argument('--groupId', type=str, default='GID_VIDEO_INVENTORY')
        self.parser.add_argument('--client_id', action='store_true')
        self.parser.add_argument('--userName', action='store_true')
        self.parser.add_argument('--password', action='store_true')
        self.parser.add_argument('--accessKey', type=str, default='LTAI4G6gf4KQpMTmMEsaf6oF')
        self.parser.add_argument('--secretkey', type=str, default='pwAzZb7DFMjju3TYI5lmcShw2nlZUs')
        self.parser.add_argument('--send', type=str, default="VIDEO_INVENTORY/VIDEO_INVENTORY_SERVER")
        self.parser.add_argument('--receive', action='store_true')

        # MQTT message
        self.parser.add_argument('--information1', action='store_true')
        self.parser.add_argument('--information2', action='store_true')
        self.parser.add_argument('--information3', action='store_true')
        self.parser.add_argument('--information4', action='store_true')
        self.parser.add_argument('--callback_vcloud_state', action='store_true')

        # streaming parameters
        self.parser.add_argument('--macAddress', default=macAddress)
        self.parser.add_argument('--rtmpIP', default=rtmpIP)
        self.parser.add_argument('--rtmpUrl', action='store_true')
        self.parser.add_argument('--camera_path', type=str)
        self.parser.add_argument('--live', type=bool, default=True)

        # vcloud upload parameters
        self.parser.add_argument('--appKey', type=str, default='0e6ddc4d212955b4896f7fe81de4e07a')
        self.parser.add_argument('--secretKey', type=str, default='93463a5a9557')
        self.parser.add_argument('--vcloudUrl', type=str, default='https://vcloud.163.com/app/vod/video/get')
        self.parser.add_argument('--httpUrl', type=str)
        self.parser.add_argument('--headers')

        # video parameters
        self.parser.add_argument('--fps', type=int)
        self.parser.add_argument('--width', type=int)
        self.parser.add_argument('--height', type=int)
        self.parser.add_argument('--original_store_path', type=str)
        self.parser.add_argument('--output_store_path', type=str)
        self.parser.add_argument('--original_temp_path', type=str)
        self.parser.add_argument('--output_temp_path', type=str)
        self.parser.add_argument('--original_auto_path', type=str)
        self.parser.add_argument('--output_auto_path', type=str)
        self.parser.add_argument('--auto_save', type=bool, default=True)
        self.parser.add_argument('--auto_save_time', type=int)

        # counter parameters
        self.parser.add_argument('--direction', type=str)

    def parse(self, args=''):
        if args == '':
            parameter = self.parser.parse_args()
        else:
            parameter = self.parser.parse_args(args)

        parameter.rtmpUrl = parameter.rtmpIP + parameter.macAddress

        parameter.httpUrl = 'http://47.104.154.17/api/inventory/WholeProcess/updateEnterInventory'
        parameter.headers = {'content-type': 'application/json'}

        parameter.receive = 'VIDEO_INVENTORY/VIDEO_INVENTORY_CUSTOM/' + parameter.macAddress
        parameter.client_id = parameter.groupId + '@@@' + parameter.macAddress
        parameter.userName = 'Signature' + '|' + parameter.accessKey + '|' + parameter.instanceId
        parameter.password = base64.b64encode(hmac.new(parameter.secretkey.encode(), parameter.client_id.encode(), sha1).digest()).decode()

        parameter.camera_path = 'rtsp://admin:guocheng660@192.168.1.64:554/Streaming/Channels/101'
        parameter.fps = 20
        parameter.width = 1280
        parameter.height = 720
        parameter.original_store_path = './video/input'
        parameter.output_store_path = './video/output'
        parameter.original_temp_path = './temp/input'
        parameter.output_temp_path = './temp/output'
        parameter.original_auto_path = './auto/input'
        parameter.output_auto_path = './auto/output'
        parameter.auto_save_time = 120

        parameter.information1 = {
            "enterCloseId": None,  # 通道id
            "deviceCode": parameter.macAddress,  # 设备code
            "sequence": None,  # 流程标识
            "deviceState": "1",  # 盘点开启1，盘点关闭0（resultType为1的时候必传，其他不传）
            "resultType": "1",  # 1盘点设备状态，0盘点结果内容
        }
        parameter.information2 = {
            "enterCloseId": None,  # 通道id
            "startTime": None,  # 通道开启时间
            "sequence": None,  # 流程标识
            "deviceCode": parameter.macAddress,  # 设备code
            "resultType": "0",  # 1盘点设备状态，0盘点结果内容
            "inventoryNumber": None,  # 盘点结果
            "uuId": None,  # uuid
        }
        parameter.information3 = {
            "enterCloseId": None,  # 通道id
            "startTime": None,  # 通道开启时间
            "sequence": None,  # 流程标识
            "deviceCode": parameter.macAddress,  # 设备code
            "resultType": "0",  # 1盘点设备状态，0盘点结果内容
            "inventoryNumber": None,  # 盘点结果
            "videoUrl": "",  # 结果视频云链接
            "uuId": None,  # uuid
            "timeLength": "",  # 结果视频时长
        }

        parameter.information4 = {
            "id": None,  # uuId
            "videoUrl": None,  # 结果视频云链接
        }

        return parameter

    def init(self, args=''):

        parameter = self.parse(args)

        return parameter
