from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import hashlib
import random
import inventory
import numpy as np
import cv2
import datetime
import socket
import os
import fcntl
import struct
import binascii
from Crypto.Cipher import AES

def get_time(flag=0):
    """
        获得日期与时间

        Parameters
        ----------
        flag : int
                返回数据格式的标记
                0 ： 返回时间和日期，格式为“YYYY-MM-DD-HH-MM-SS-FFFFFF”
                1 ： 返回日期，格式为“YYYY-MM-DD”
                2 ： 返回时间，格式为“HH-MM-SS-FFFFFF”

        Returns
        -------
        return : str
                依据flag返回对应数据
    """
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

        # 定义秘钥信息
        self.seperateKey = "505505@home"
        self.aesKey = "11223344aabbccdd"
        self.aesIv = "55667788eeffgghh"
        self.aesMode = AES.MODE_CBC

    def getHwAddr(self, ifname):
        """
        获取主机物理地址
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        info = fcntl.ioctl(s.fileno(), 0x8927, struct.pack('256s', bytes(ifname[:15], 'utf-8')))
        return ''.join(['%02x' % char for char in info[18:24]])

    def decrypt(self, text):
        """
        从license文件中解密出主机地址
        """
        try:
            cryptor = AES.new(self.aesKey.encode('utf-8'), self.aesMode, self.aesIv.encode('utf-8'))
            plain_text = cryptor.decrypt(binascii.unhexlify(text))
            return plain_text
        except:
            return ""

    def getLicenseInfo(self, filePath=None):
        if filePath == None:
            filePath = "./license.lic"  # 留个坑没写传参的部分，感觉没太大必要

        if not os.path.isfile(filePath):
            print("请将 license.lic 文件放在当前路径下")
            os._exit(0)
            return False, 'Invalid'

        encryptText = ""
        with open(filePath, "rb") as licFile:
            encryptText = licFile.read()
            licFile.close()
        try:
            hostInfo = self.getHwAddr('enp3s0')

        except IOError:
            hostInfo = self.getHwAddr('wlp4s0')

        decryptText = self.decrypt(encryptText)
        pos = decryptText.decode().find(self.seperateKey)
        if -1 == pos:
            return False, "Invalid"
        licHostInfo = self.decrypt((decryptText[0:pos]).decode())
        licHostInfo = licHostInfo.decode().rstrip('7')
        true_licHostInfo = licHostInfo.split('-')[0]
        licenseStr = decryptText[pos + len(self.seperateKey):]

        date =  get_time(1)
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


def get_md5(data):
    """
        计算数据的md5

        Parameters
        ----------
        data : all type
                需要计算md5的数据块

        Returns
        -------
        return : str
                输入数据块的md5值(128位)
    """
    myhash = hashlib.md5()
    myhash.update(data)
    return myhash.hexdigest()



def recv_all(sock, count):
    """
        为解决某些机器一次性接收图像接收不全的问题，采用分批次接收的方法

        Parameters
        ----------
        sock : sock_
                通信句柄
        count : int
                图片数据长度

        Returns
        -------
        buf : bytes
                收到的图片数据
    """
    buf = b''
    while count:
        if count > 1024:
            newbuf = sock.recv(1024)
        else:
            newbuf = sock.recv(count)
        if not newbuf:
            return
        buf += newbuf
        count -= len(newbuf)
    return buf


def mask_pic(img, mask, pigNumber):
    """
        为图像涂掩模

        Parameters
        ----------
        img : cv2.Mat/np.array
                需要涂抹掩模的图像
        mask : cv2.Mat/np.array
                掩模(形状必须与img一致)
        pigNumber : int
                掩模中猪只数目

        Returns
        -------
        img : cv2.Mat/np.array
                涂抹掩模后的图像
    """
    if img.shape != mask.shape:
        pass
    else:
        for i in range(pigNumber):
            threshold = i + 1
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            img[..., 0] = np.where(mask == threshold, b, img[..., 0])
            img[..., 1] = np.where(mask == threshold, g, img[..., 1])
            img[..., 2] = np.where(mask == threshold, r, img[..., 2])
    return img


class socket_client():
    def __init__(self, config):
        """
            类初始化

            Parameters
            ----------
            config : class of configparser
                    config的读写类

            Returns
            -------
            None
        """
        self.config = config
        self.model = inventory.model(self.config.get('maskRcnnInit', 'gpu_select'), float(self.config.get('maskRcnnInit', 'gpu_menmory')), model_path=self.config.get('model', 'model_path'), equipment_key=self.config.get('model', 'equipment_key'), usr_key=self.config.get('model', 'equipment_key'))

        self.addr = (self.config.get('network', 'ip'), int(self.config.get('network', 'port')))


    def login(self):
        """
            登录集群服务器
            登录格式为“login^_^name^_^task^_^version^_^type^_^000\n”

            Returns
            -------
            True or raise Exception
        """
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect(self.addr)

        sendmsg = 'login^_^' + self.config.get('workstation', 'name') + '^_^' \
              + self.config.get('workstation', 'task') + '^_^' \
              + self.config.get('workstation', 'ver') + '^_^' \
              + self.config.get('workstation', 'type') + '^_^' \
              + '000\n'
        self.conn.send(bytes(sendmsg, encoding='utf-8'))

        backmsg = self.conn.recv(128).decode()
        if backmsg.split('^_^')[1] == 'success':
            print('{time} connect to Cluster success'.format(time=get_time(2)))
            return True
        else:
            raise Exception('login error')


    def recv_imgInfo(self):
        """
            请求图像信息
            发送格式：“task^_^ask^_^000\n”
            接收格式：“imgLength^_^picLength^_^md5^_^000\n”
            返回格式：“imgLength^_^success^_^000\n”

            Returns
            -------
            True or raise Exception
        """
        self.conn.send(bytes('{}^_^ask^_^000\n'.format(self.config.get('workstation', 'task')), encoding='utf-8'))
        recv = self.conn.recv(128).decode().split('^_^')
        if len(recv) == 4 and recv[0] == 'imgLength':
            self.picLength = int(recv[1])
            self.md5_true = recv[2]
            self.conn.send(bytes('imgLength^_^success^_^000\n', encoding='utf-8'))
            print('{time} Received imgLength = {length}'.format(time=get_time(2),length=self.picLength))
            return True
        else:
            raise Exception('imgInfo error')

    def recv_picture(self):
        """
            请求图像，并校验md5
            接收格式：图像信息(bytes)

            Returns
            -------
            True or raise Exception
        """
        self.stringData0 = recv_all(self.conn, self.picLength)
        if (self.md5_true == get_md5(self.stringData0)):
            print('{time} Received image success'.format(time=get_time(2)))
            return True
        else:
            raise Exception('image error')

    def get_result(self):
        """
            调用模型计算结果并返回
            发送格式：“numResult^_^numResult^_^picLength^_^picMD5^_^000\n”
            接收格式：“numResult^_^success^_^000\n”

            Returns
            -------
            True or raise Exception
        """
        data = np.fromstring(self.stringData0, dtype='uint8')
        img = cv2.imdecode(data, 1)
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.config.get('workstation', 'type') == '4':
            if img_RGB.shape == (1080, 1920, 3):
                decimg = img_RGB[0:1080, 773:1257]
                assert lic_match()
                results = self.model.getMaskResultFromNumpy(image=decimg)
                result = results['sum']
                if results['sum'] == 0:
                    pass
                else:
                    bias = 773
                    box = results['rois'][0]
                    cv2.rectangle(img, (box[1] + bias, box[0]), (box[3] + bias, box[2]), (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)), 4)
            else:
                result = -1
            _, resultimg = cv2.imencode('.png', img)
            resultdata = np.array(resultimg)
        else:
            result = -2    # type error
            _, resultimg = cv2.imencode('.png', img)
            resultdata = np.array(resultimg)


        self.stringData1 = resultdata.tostring()
        sendback = 'numResult^_^{num}^_^{length}^_^{md5}^_^000\n'.format(num=result,
                                                                           length=(len(self.stringData1)),
                                                                           md5=get_md5(self.stringData1))
        print('{time}: is_pig={num} proImgLength={length}'.format(time=get_time(2),
                                                                  num=result,
                                                                  length=(len(self.stringData1))))
        self.conn.send(bytes(sendback, encoding='utf-8'))
        backmsg = self.conn.recv(128).decode().split('^_^')
        if len(backmsg) == 3 and backmsg[1] == 'success':
            print('{time} numReault send success'.format(time=get_time(2)))
            return True
        else:
            raise Exception('numResult send error')

    def send_picture(self):
        """
            发送带掩模/框的结果图像
            发送格式：图像信息(bytes)
            接收格式：“proImg^_^success^_^000\n”

            Returns
            -------
            True or raise Exception
        """
        self.conn.send(self.stringData1)
        backmsg = self.conn.recv(128).decode().split('^_^')
        if len(backmsg) == 3 and backmsg[1] == 'success':
            print('{time} Picture send success'.format(time=get_time(2)))
            return True
        else:
            raise Exception('picture send error')

    def error_handle(self, error):
        """
            错误处理函数，支持错误打印并重连

            Parameters
            ----------
            error : Exception
                    错误信息

            Returns
            -------
            None
        """
        print('{time} {e}'.format(time=get_time(2), e=error))
        time.sleep(5)
        self.run()

    def run(self):
        try:
            self.login()
            while True:
                self.recv_imgInfo()
                self.recv_picture()
                self.get_result()
                self.send_picture()
        except Exception as e:
            self.error_handle(e)
