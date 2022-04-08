"""
External Version 2.0
Internal Version 3.0
2021/04/30 by Joshua Wem & BWH from HEU & XiaoLong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy
import hmac
import base64
from json import dumps
from hashlib import sha1
import paho.mqtt.client as mqtt


class RpcClient(object):
    def __init__(self, share_var, share_var_1, share_var_2, share_var_3, share_var_4, share_lock, share_lock_1, share_lock_2, share_lock_3, share_lock_4, parameters):
        self.share_var = share_var
        self.share_var_1 = share_var_1
        self.share_var_2 = share_var_2
        self.share_var_3 = share_var_3
        self.share_var_4 = share_var_4
        self.share_lock = share_lock
        self.share_lock_1 = share_lock_1
        self.share_lock_2 = share_lock_2
        self.share_lock_3 = share_lock_3
        self.share_lock_4 = share_lock_4

        self.p = parameters

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected with result code " + str(rc))
            print("连接成功")
            client.subscribe(topic=self.p.receive)
        else:
            print("连接失败")

    def on_disconnect(self, client, userdata, rc):
        if rc == 1:
            print("协议版本错误")
        elif rc == 2:
            print("无效的客户端标识")
        elif rc == 3:
            print("服务器无法使用")
        elif rc == 4:
            print("错误的用户名或密码")
        elif rc == 5:
            print("未经授权")

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("等待消息接收...")

    def on_publish(self, client, userdata, mid):
        print("消息发送成功")

    def on_message(self, client, userdata, msg):

        print(f"\nReceived `{msg.payload.decode()}` from `{msg.topic}` topic")

        if eval(msg.payload.decode())["videoState"] == "1":
            message_1 = {}
            message_1["uuId"] = eval(msg.payload.decode())["uuId"]
            self.share_lock_1.acquire()
            self.share_var_1[:] = []
            self.share_var_1.append(message_1)
            self.share_lock_1.release()
            blog = 1
            self.share_lock.acquire()
            self.share_var.set(blog)
            self.share_lock.release()

            self.p.information1["enterCloseId"] = eval(msg.payload.decode())["enterCloseId"]
            self.p.information1["sequence"] = eval(msg.payload.decode())["sequence"]
            self.p.information2["startTime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            client.publish(topic=self.p.send, payload=dumps(self.p.information1))
            print('Send terminal message: ', dumps(self.p.information1))

        elif eval(msg.payload.decode())["videoState"] == "0":
            blog = 0
            self.share_lock.acquire()
            self.share_var.set(blog)
            self.share_lock.release()

            self.p.information2["enterCloseId"] = eval(msg.payload.decode())["enterCloseId"]
            self.p.information2["uuId"] = eval(msg.payload.decode())["uuId"]
            # self.p.information2["startTime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.p.information2["sequence"] = eval(msg.payload.decode())["sequence"]

            time.sleep(2)
            self.share_lock_4.acquire()
            message_2 = copy.copy(self.share_var_4)
            self.share_lock_4.release()
            print(message_2)
            self.p.information2["inventoryNumber"] = message_2[0]["total_number"]
            # self.p.information3["inventoryNumber"] = message_2[0]["total_number"]
            client.publish(topic=self.p.send, payload=dumps(self.p.information2))
            print('Send terminal message: ', dumps(self.p.information2))

        # elif eval(msg.payload.decode())["videoState"] == "2":
        #     blog = 2
        #     self.upload_flag = 0
        #     self.share_lock.acquire()
        #     self.share_var.set(blog)
        #     self.share_lock.release()
        #
        #     self.share_lock_1.acquire()
        #     message = copy.copy(self.share_var_1)
        #     self.share_lock_1.release()
        #
        #     while not self.upload_flag:
        #         self.share_lock_3.acquire()
        #         self.upload_flag = self.share_var_3.get()
        #         self.share_lock_3.release()
        #
        #     blog = 3
        #     self.share_lock.acquire()
        #     self.share_var.set(blog)
        #     self.share_lock.release()
        #
        #     if self.upload_flag == 1:
        #         self.share_lock_2.acquire()
        #         url = copy.copy(self.share_var_2)
        #         self.share_lock_2.release()
        #
        #         self.p.information3["enterCloseId"] = eval(msg.payload.decode())["enterCloseId"]
        #         self.p.information3["startTime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #         self.p.information3["sequence"] = eval(msg.payload.decode())["sequence"]
        #         self.p.information3["uuId"] = eval(msg.payload.decode())["uuId"]
        #         self.p.information3["timeLength"] = message[0]["timeLength"]
        #         self.p.information3["videoUrl"] = url[0]
        #         client.publish(topic=self.p.send, payload=dumps(self.p.information3))
        #         print('Send terminal message: ', dumps(self.p.information3))

    def run(self):
        client = mqtt.Client(self.p.client_id, protocol=mqtt.MQTTv311, clean_session=True)
        client.on_connect = self.on_connect
        client.on_disconnect = self.on_disconnect
        client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message
        client.on_publish = self.on_publish
        userName = 'Signature' + '|' + self.p.accessKey + '|' + self.p.instanceId
        password = base64.b64encode(hmac.new(self.p.secretkey.encode(), self.p.client_id.encode(), sha1).digest()).decode()
        client.username_pw_set(userName, password)
        client.connect(self.p.host, self.p.port, self.p.keepalive)
        client.loop_forever()
