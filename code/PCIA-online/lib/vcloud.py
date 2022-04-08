from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
import json
import hashlib
import random
import time
import os
import base64
import tempfile

VCLOUD_API_HOST = 'https://vcloud.163.com'  # 视频云API Host

WANPROXY_HOST = 'http://wanproxy.127.net'  # 获取上传加速节点 Host

_BLOCK_SIZE = 1024 * 1024 * 4  # 断点续上传分块大小，该参数为接口规格，暂不支持修改

POOL_CONNECTIONS = 10  # 链接池个数为10
CONNECTION_TIMEOUT = 30  # 链接超时为时间为30s
CONNECTION_RETRIES = 3  # 链接重试次数为3次

TMP_FILE = './video/output'  # 用于存储断点的临时文件目录，一个上传文件会对应一个临时文件，上传完成会清除临时文件，应保证对此目录有读写权限

_session = None


# UTILS
def _file_iter(input_stream, size, offset=0):
    """读取输入流:

    Args:
        input_stream: 待读取文件的二进制流
        size:         二进制流的大小

    Raises:
        IOError: 文件流读取失败
    """
    input_stream.seek(offset)
    d = input_stream.read(size)
    while d:
        yield d
        d = input_stream.read(size)


# HTTP
def _return_wrapper(resp):
    if resp.status_code != 200:
        return None, ResponseInfo(resp)
    ret = resp.json() if resp.text != '' else {}
    return ret, ResponseInfo(resp)


def _init():
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=POOL_CONNECTIONS,
                                            pool_maxsize=POOL_CONNECTIONS, max_retries=CONNECTION_RETRIES)
    session.mount('http://', adapter)
    global _session
    _session = session


def _get(url, body=None, headers={}):
    if _session is None:
        _init()
    try:
        r = _session.get(url, headers=headers)
    except Exception as e:
        print("_get exception : {0}".format(e))
        return None, ResponseInfo(None, e)
    return _return_wrapper(r)


def _post_json(url, body=None, headers={}):
    if _session is None:
        _init()
    try:
        r = _session.post(url, json=body, headers=headers, timeout=CONNECTION_TIMEOUT)
    except Exception as e:
        return None, ResponseInfo(None, e)
    return _return_wrapper(r)


def _post(url, headers, stream):
    if _session is None:
        _init()
    try:
        r = _session.post(url, headers=headers, data=stream)
    except Exception as e:
        return None, ResponseInfo(None, e)
    return _return_wrapper(r)


def _post_file(url, headers, stream):
    return _post(url, headers, stream)


class ResponseInfo(object):
    """HTTP请求返回信息类

	该类主要是用于获取和解析对视频云发起各种请求后的响应包的header和body。

	Attributes:
		status_code: 整数变量，响应状态码
		text_body:   字符串变量，响应的body
		error:       字符串变量，响应的错误内容
    """

    def __init__(self, response, exception=None):
        """用响应包和异常信息初始化ResponseInfo类"""
        self.response = response
        self.exception = exception
        if response is None:
            self.status_code = -1
            self.text_body = None
            self.error = str(exception)
        else:
            self.status_code = response.status_code
            self.text_body = response.text

    def ok(self):
        return self.status_code == 200

    def need_retry(self):
        if self.response is None:
            return True
        code = self.status_code
        if code // 100 == 5:
            return True
        return False

    def connect_failed(self):
        return self.response is None

    def __str__(self):
        return ', '.join(['%s:%s' % item for item in self.__dict__.items()])

    def __repr__(self):
        return self.__str__()


# TRANSPORT
class Transport(object):
    """视频云请求实体类

    该类主要实现了视频云的相关请求

    Attributes:
        access_key_id:		用户的访问密钥公钥
        access_key_secret:	用户的访问密钥私钥
    """

    def __init__(self, access_key_id=None, access_key_secret=None):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret

    def vcloud_api_request(self, operator, headers={}, body=None):
        """视频云API访问
        Args:
            operator:	API方法
            headers:	请求头
            body:		请求包体

        Returns:
            一个dict变量，类似 {"code": "<code>", "ret": {"<ret>"}}
            一个ResponseInfo对象
        """
        url = VCLOUD_API_HOST + operator
        r, info = _post_json(url, body, headers)

        if info.status_code == 200:
            data = json.loads(info.text_body)
            if data["code"] != 200:
                return r, VcloudResp(data["code"], None, data["msg"])
            else:
                return r, VcloudResp(data["code"], data["ret"], None)
        return r, None

    def upload_host_get(self):
        """获取上传加速节点
        Returns:
            一个dict变量
            一个ResponseInfo对象
        """
        url = WANPROXY_HOST + "/lbs?version=1.0"
        ret, info = _get(url)

        return ret, info

    def get_offset(self, upload_host, upload_init, context):
        """获取断点
        Args:
            upload_host:	上传加速节点列表
            upload_init:	上传初始化信息
            context:		标示断点的上下文

        Returns:
            文件下一次上传的起始偏移量
        """
        url = upload_host.upload_host + "/" + upload_init.bucket + "/" + upload_init.objectName + "?uploadContext&context=" + context + "&version=1.0"
        headers = {}
        headers['x-nos-token'] = upload_init.xNosToken
        ret, info = _get(url, None, headers)

        if ret is None and not info.need_retry():
            return 0
        if info.connect_failed():
            host_backup = upload_host.upload_host_backup
            url_backup = host_backup + "/" + upload_init.bucket + "/" + upload_init.objectName + "?uploadContext&context=" + context + "&version=1.0"
            if info.need_retry():
                ret, info = _get(url_backup, None, headers)
                if ret is None:
                    return 0

        if info.status_code == 200:
            data = json.loads(info.text_body)
            return int(data["offset"])
        else:
            return 0

    def upload_file(self, upload_init, upload_host, file_name, file_path, file_size, modify_time, offset, context,
                    upload_progress_recorder, mime_type='application/octet-stream', progress_handler=None):
        """上传文件
        Args:
            upload_init:				上传初始化信息
            upload_host:				上传加速节点列表
            file_name:					上传文件名
            file_path:					上传文件的路径
            file_size:					上传文件大小
            modify_time:				上传文件最后修改时间
            offset:						文件下一次上传的起始偏移量
            context:					标示断点的上下文
            upload_progress_recorder:	记录上传进度，用于断点续传
            mime_type:					上传数据的mimeType
            progress_handler:			上传进度

        Returns:
            一个dict变量，类似 {"code": "<code>", "ret": {"<ret>"}}
            一个ResponseInfo对象
        """
        return put_file(upload_init, upload_host, file_name, file_path, file_size, modify_time, offset, context,
                        upload_progress_recorder, mime_type, progress_handler)


# STORAGE
class UploadProgressRecorder(object):
    """持久化上传记录类

    该类默认保存每个文件的上传记录到文件系统中，用于断点续传
    上传记录为json格式：
    {
        "bucket": bucket,
        "object": object,
        "xNosToken": x-nos-token,
        "size": file_size,
        "context": context,
        "host": upload_host
    }

    Attributes:
        record_folder: 保存上传记录的目录
    """
    def __init__(self, record_folder=tempfile.gettempdir()):
        self.record_folder = record_folder

    def get_upload_record(self, file_name, key):
        key = '{0}/{1}'.format(key, file_name)
        record_file_name = base64.b64encode(key.encode('utf-8')).decode('utf-8')
        upload_record_file_path = os.path.join(self.record_folder, record_file_name)

        if not os.path.isfile(upload_record_file_path):
            return None
        with open(upload_record_file_path, 'r') as f:
            json_data = json.load(f)
        return json_data

    def set_upload_record(self, file_name, key, data):
        key = '{0}/{1}'.format(key, file_name)
        record_file_name = base64.b64encode(key.encode('utf-8')).decode('utf-8')
        upload_record_file_path = os.path.join(self.record_folder, record_file_name)
        with open(upload_record_file_path, 'w') as f:
            json.dump(data, f)

    def delete_upload_record(self, file_name, key):
        key = '{0}/{1}'.format(key, file_name)
        record_file_name = base64.b64encode(key.encode('utf-8')).decode('utf-8')
        record_file_path = os.path.join(self.record_folder, record_file_name)
        os.remove(record_file_path)


def put_file(upload_init, upload_host, file_name, file_path, file_size, modify_time, offset, context,
             upload_progress_recorder, mime_type='application/octet-stream', progress_handler=None):
    """上传文件
	Args:
		upload_init:				上传初始化信息
		upload_host:				上传加速节点列表
		file_name:					上传文件名
		file_path:					上传文件的路径
		file_size:					上传文件大小
		modify_time:				上传文件最后修改时间
		offset:						文件下一次上传的起始偏移量
		context:					标示断点的上下文
		upload_progress_recorder:	记录上传进度，用于断点续传
		mime_type:					上传数据的mimeType
		progress_handler:			上传进度

	Returns:
		一个dict变量，类似 {"code": "<code>", "ret": {"<ret>"}}
		一个ResponseInfo对象
	"""
    ret = {}
    info = None
    with open(file_path, 'rb') as input_stream:
        ret, info = put_stream(upload_init, upload_host, file_name, file_size, modify_time, offset, context,
                               input_stream, upload_progress_recorder, mime_type, progress_handler)
    return ret, info


def put_stream(upload_init, upload_host, file_name, size, modify_time, offset, context, input_stream,
               upload_progress_recorder, mime_type, progress_handler=None):
    task = _Resume(upload_init, upload_host, file_name, size, modify_time, offset, context, input_stream,
                   upload_progress_recorder, mime_type, progress_handler)
    return task.upload()


class _Resume(object):
    """
    断点续上传类
	该类主要实现了分块上传，断点续上

	Attributes:
		upload_init:				上传初始化信息
		upload_host:				上传加速节点列表
		file_name:					上传文件名
		size:						上传文件大小
		modify_time:				上传文件最后修改时间
		offset:						文件下一次上传的起始偏移量
		context:					标示断点的上下文
		input_stream:				上传二进制流
		upload_progress_recorder:	记录上传进度，用于断点续传
		mime_type:					上传数据的mimeType
		progress_handler:			上传进度
	"""

    def __init__(self, upload_init, upload_host, file_name, size, modify_time, offset, context, input_stream,
                 upload_progress_recorder, mime_type, progress_handler):
        """初始化断点续上传"""
        self.upload_init = upload_init
        self.upload_host = upload_host
        self.file_name = file_name
        self.offset = offset
        self.context = context
        self.size = size
        self.modify_time = modify_time
        self.input_stream = input_stream
        self.mime_type = mime_type
        self.upload_progress_recorder = upload_progress_recorder
        self.progress_handler = progress_handler

    def upload(self):
        """上传操作"""
        ret = None
        info = None

        host = self.upload_host.upload_host
        offset = self.offset
        context = self.context
        record = False

        for block in _file_iter(self.input_stream, _BLOCK_SIZE, offset):
            url = self.block_url(host, offset, context)
            ret_inner, info_inner = self.block_upload(url, block)

            if ret_inner is None and not info_inner.need_retry():
                return ret_inner, info_inner
            if info_inner.connect_failed():
                host_backup = self.upload_host.upload_host_backup
                url_backup = self.block_url(host_backup, offset, context)
                if info_inner.need_retry():
                    ret_inner, info_inner = self.block_upload(url_backup, block)
                    if ret_inner is None:
                        return ret_inner, info_inner

            data = json.loads(info_inner.text_body)
            if "offset" in data:
                offset = data["offset"]
            if "context" in data:
                context = data["context"]

            if not record:
                self.record_upload_progress(context)
                record = True

            ret = ret_inner
            info = info_inner
            if (callable(self.progress_handler)):
                self.progress_handler(offset, self.size)
        # 进度显示
        # print "upload process ... {0}".format(self.as_parent(offset, self.size))
        self.upload_progress_recorder.delete_upload_record(self.file_name, self.modify_time)
        return ret, info

    def block_upload(self, url, block):
        """上传块"""
        headers = {}
        headers['x-nos-token'] = self.upload_init.xNosToken
        return _post(url, headers, block)

    def block_url(self, host, offset, context):
        """获取上传块的URL"""
        if self.size - offset < _BLOCK_SIZE:
            if offset == 0:
                return '{0}/{1}/{2}?offset={3}&complete={4}&version=1.0'.format(host, self.upload_init.bucket,
                                                                                self.upload_init.objectName, offset,
                                                                                'true')
            else:
                return '{0}/{1}/{2}?offset={3}&complete={4}&context={5}&version=1.0'.format(host,
                                                                                            self.upload_init.bucket,
                                                                                            self.upload_init.objectName,
                                                                                            offset, 'true', context)
        else:
            return '{0}/{1}/{2}?offset={3}&complete={4}&context={5}&version=1.0'.format(host, self.upload_init.bucket,
                                                                                        self.upload_init.objectName,
                                                                                        offset, 'false', context)

    def record_upload_progress(self, context):
        """记录上传的断点信息"""
        record_data = {
            'bucket': self.upload_init.bucket,
            'object': self.upload_init.objectName,
            'xNosToken': self.upload_init.xNosToken,
            'size': self.size,
            'context': context,
            'host': vars(self.upload_host)
        }
        self.upload_progress_recorder.set_upload_record(self.file_name, self.modify_time, record_data)

    def as_parent(self, num, den):
        if den == 0:
            ratio = 0
        else:
            ratio = float(num) / den
        return "%5.1f%%" % (100 * ratio)


# MODEL
class VcloudResp(object):
    """
    视频云API返回结果封装类该类主要封装了视频云API放回结果
	Attributes:
		code:	视频云API返回码
		ret:	视频云API返回结果
		msg:	视频云API返回错误信息
	"""

    def __init__(self, code, ret, msg):
        self.code = code
        self.ret = ret
        self.msg = msg


class UploadInit(object):
    """
	上传初始化信息封装类
	该类主要封装了上传初始化信息

	Attributes:
		bucket:		存储上传文件的桶
		objectName:	上传文件存储的名称
		xNosToken:	访问上传加速节点的上传凭证
	"""

    def __init__(self, bucket, objectName, xNosToken):
        self.bucket = bucket
        self.objectName = objectName
        self.xNosToken = xNosToken


class QueryResult(object):
    """查询上传文件ID返回信息类

	该类主要封装了上传文件完成后返回的ID信息

	Attributes:
		objectName:		上传文件的保存名称，即objectName
		vid:			上传视频文件返回的ID
		imgId:			上传图片文件返回的ID
	"""

    def __init__(self, objectName, vid, imgId):
        self.objectName = objectName
        self.vid = vid
        self.imgId = imgId


class UploadHost(object):
    """
    查询上传加速节点返回信息类
	该类主要封装了上传加速节点列表

	Attributes:
		upload_host:		上传加速节点首选节点
		upload_host_backup:	上传加速节点备用节点
	"""

    def __init__(self, upload_host, upload_host_backup):
        self.upload_host = upload_host
        self.upload_host_backup = upload_host_backup


# CLIENT
class Auth(object):
    """
    视频云权限类
	该类主要实现了视频云的请求权限获取

	Attributes:
		AppKey:		用户的访问密钥公钥
		AppSecret:	用户的访问密钥私钥
	"""

    def __init__(self, AppKey, AppSecret):
        self.AppKey = AppKey
        self.AppSecret = AppSecret
        charHex = '0123456789abcdef';
        self.Nonce = '';  # 随机字符串最大128个字符，也可以小于该数
        for i in range(0, 128):
            index = int(15 * random.random());
            self.Nonce = self.Nonce + charHex[index];

        self.CurTime = int(time.time());  # 当前UTC时间戳，从1970年1月1日0点0 分0 秒开始到现在的秒数(String)

    def checkSumBuilder(self):
        """获取CheckSum
		"""
        join_string = self.AppSecret + self.Nonce + str(self.CurTime);
        self.CheckSum = hashlib.sha1(join_string.encode(
            'utf-8')).hexdigest();  # SHA1(AppSecret + Nonce + CurTime),三个参数拼接的字符串，进行SHA1哈希计算，转化成16进制字符(String，小写)

    def getVcloudHeaders(self):
        """获取视频云API的请求头
		Returns:
			 一个dict变量，视频云请求头
		"""
        self.checkSumBuilder()
        headers = {}
        headers['Content-Type'] = 'application/json;charset=utf-8'
        headers['AppKey'] = self.AppKey
        headers['Nonce'] = self.Nonce
        headers['CurTime'] = str(self.CurTime)
        headers['CheckSum'] = self.CheckSum
        return headers


class Client(object):
    """视频云请求实体类

	该类主要实现了视频云的相关请求

	Attributes:
		access_key_id:		用户的访问密钥公钥
		access_key_secret:	用户的访问密钥私钥
		transport_class:	http请求处理类
	"""

    def __init__(self, access_key_id, access_key_secret, transport_class=Transport):
        self.transport = transport_class()
        self.auth = Auth(access_key_id, access_key_secret)

    def upload_init(self, body):
        """上传初始化
		Args:
			body:	请求包体

		Returns:
			一个UploadInit对象
		"""
        headers = self.auth.getVcloudHeaders()
        r, vResp = self.transport.vcloud_api_request("/app/vod/upload/init", headers, body)
        if vResp is not None:
            if vResp.msg is None:
                return UploadInit(vResp.ret['bucket'], vResp.ret['object'], vResp.ret['xNosToken'])
            else:
                return None
        else:
            return None

    def get_upload_host(self):
        """获取上传加速节点地址
		Returns:
			一个UploadHost对象
		"""
        ret, info = self.transport.upload_host_get()
        if info is not None:
            if info.status_code == 200:
                data = json.loads(info.text_body)
                return UploadHost(data['upload'][0], data['upload'][1])
            else:
                return None
        else:
            return None

    def upload_file(self, body, file_path, progress_handler=None, mime_type='application/octet-stream'):
        """上传文件
		Args:
			body:		请求包体
			file_path:	上传文件路径

		Returns:
			一个QueryResult对象
		"""
        file_name = os.path.basename(file_path)
        modify_time = os.path.getmtime(file_path)
        file_size = os.stat(file_path).st_size

        upload_init = None
        upload_host = None
        offset = 0
        context = ""

        upload_progress_recorder = UploadProgressRecorder(TMP_FILE)
        record = upload_progress_recorder.get_upload_record(file_name, modify_time)
        if record is None:
            upload_init = self.upload_init(body)
            upload_host = self.get_upload_host()
        else:
            upload_init = UploadInit(record['bucket'], record['object'], record['xNosToken'])
            upload_host = UploadHost(record['host']['upload_host'], record['host']['upload_host_backup'])
            context = record['context']
            offset = self.transport.get_offset(upload_host, upload_init, context)

        if upload_init is None or upload_host is None:
            return None

        ret, info = self.transport.upload_file(upload_init, upload_host, file_name, file_path, file_size, modify_time,
                                               offset, context, upload_progress_recorder, mime_type, progress_handler)

        if info.status_code == 200:
            body = {"objectNames": [upload_init.objectName]}
            return self.query_id(body)
        else:
            return None

    def query_id(self, body):
        """根据对象名查询uid或imgId
		Args:
			body:	请求包体

		Returns:
			一个QueryResult对象
		"""
        headers = self.auth.getVcloudHeaders()
        r, vResp = self.transport.vcloud_api_request("/app/vod/video/query", headers, body)
        if vResp is not None:
            if vResp.msg is None:
                result = vResp.ret['list'][0]
                if 'vid' in result:
                    return QueryResult(result['objectName'], result['vid'], None)
                else:
                    return QueryResult(result['objectName'], None, result['imgId'])
            else:
                return vResp.msg.encode('utf-8')
        else:
            return "请求异常"

    def set_callback(self, body):
        """设置上传回调
		Args:
			body:	请求包体

		Returns:
			字符串，表示请求结果
		"""
        headers = self.auth.getVcloudHeaders()
        r, vResp = self.transport.vcloud_api_request("/app/vod/upload/setcallback", headers, body)

        if vResp is not None:
            if vResp.code == 200:
                return "设置成功"
            else:
                return vResp.msg.encode('utf-8')
        else:
            return "请求异常"

    def get_url(self, body):
        headers = self.auth.getVcloudHeaders()
        r, vResp = self.transport.vcloud_api_request("/app/vod/video/get", headers, body)
        if vResp is not None:
            if vResp.code == 200:
                return "设置成功", r
            else:
                return vResp.msg.encode('utf-8')
        else:
            return "请求异常"
