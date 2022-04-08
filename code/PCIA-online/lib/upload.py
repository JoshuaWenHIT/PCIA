"""
External Version 1.0
Internal Version 2.0
2021/04/30 by Joshua Wen from HEU & XiaoLong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import requests
from json import dumps
from lib.vcloud import Client


def upload_progress(offset, size):
    if size == 0:
        print("\rupload process ... {0}".format("%5.1f%%" % (100 * 0)), end='')
    else:
        ratio = float(offset) / size
        print("\rupload process ... {0}".format("%5.1f%%" % (100 * ratio)), end='')


def upload(appKey, secretKey, parameters):
    p = parameters
    client = Client(appKey, secretKey)
    path = get_latest_file(p.output_store_path)
    body1 = {"originFileName": path.split('/')[-1]}
    print('The file will be uploaded: ', os.path.join(p.output_store_path, body1['originFileName']))
    print('\n############################## 开始视频上传 ###############################')
    res = client.upload_file(body1, path, upload_progress)  # 上传接口
    if res is not None:
        print("\n上传回调内容: {0}".format(vars(res)))
    else:
        print("\n上传回调内容为空，上传出错 !")
    vid = vars(res)['vid']
    body2 = {"vid": vid}
    res2 = client.get_url(body2)
    return res2


def upload_v2(appKey, secretKey, parameters):
    p = parameters
    client = Client(appKey, secretKey)
    NU_list, Uing_list, Ued_list = get_file_state(p.output_store_path)
    results = []
    for filename in NU_list:
        uuId = filename.split('.')[0].split('_')[-1]
        body1 = {"originFileName": filename}
        print('\n############################## 开始视频上传 ###############################')
        print('The file will be uploaded: ', os.path.join(p.output_store_path, body1['originFileName']))
        res = client.upload_file(body1, os.path.join(p.output_store_path, body1['originFileName']), upload_progress)
        if res is not None:
            print("\n上传回调内容: {0}".format(vars(res)))
        else:
            print("\n上传回调内容为空，上传出错 !")
        vid = vars(res)['vid']
        body2 = {"vid": vid}
        res2 = client.get_url(body2)
        # results.append((uuId, res2[1]['ret']['origUrl'], filename))
        p.information4["id"] = uuId
        p.information4["videoUrl"] = res2[1]['ret']['origUrl']
        print(dumps(p.information4))
        ret = requests.post(p.httpUrl, json=json.loads(dumps(p.information4)), headers=p.headers)
        # if ret.status_code == 200:
        #     rename(os.path.join(p.output_store_path, filename), p, state='NU2Ued')
        #     print(json.loads(ret.text))
        if json.loads(ret.text)['code'] == '0':
            rename(os.path.join(p.output_store_path, filename), p, state='NU2Ued')
            print(json.loads(ret.text))
        else:
            print("接口请求失败 !")
            print(json.loads(ret.text))
    print('\n############################## 视频上传完毕 ###############################')
    return results


def get_latest_file(dir_path):
    lists = os.listdir(dir_path)
    lists.sort(key=lambda fn: os.path.getmtime(dir_path + '/' + fn))
    file_latest = os.path.join(dir_path, lists[-1])
    return file_latest


def get_file_state(dir_path):
    NU_list = []
    Uing_list = []
    Ued_list = []
    for filename in os.listdir(dir_path):
        if filename.split('_')[0] == 'NU':
            NU_list.append(filename)
        elif filename.split('_')[0] == 'Uing':
            Uing_list.append(filename)
        elif filename.split('_')[0] == 'Ued':
            Ued_list.append(filename)
    return NU_list, Uing_list, Ued_list


def rename(file, parameters, state):
    p = parameters
    filename = str(file.split('/')[-1])
    if state == 'NU2Uing':
        os.rename(file, os.path.join(p.output_store_path, filename.replace('NU', 'Uing')))
        return os.path.join(p.output_store_path, filename.replace('NU', 'Uing'))
    elif state == 'Uing2Ued':
        os.rename(file, os.path.join(p.output_store_path, filename.replace('Uing', 'Ued')))
        return os.path.join(p.output_store_path, filename.replace('Uing', 'Ued'))
    elif state == 'NU2Ued':
        os.rename(file, os.path.join(p.output_store_path, filename.replace('NU', 'Ued')))
        return os.path.join(p.output_store_path, filename.replace('NU', 'Ued'))


def start_upload(share_var, share_var_2, share_var_3, share_lock, share_lock_2, share_lock_3, parameters):
    p = parameters
    upload_blog = False
    while True:
        share_lock.acquire()
        blog = share_var.get()
        share_lock.release()

        if blog == 2:
            if not upload_blog:
                res2 = upload(p.appKey, p.secretKey, p)
                url = res2[1]['ret']['origUrl']

                share_lock_2.acquire()
                share_var_2[:] = []
                share_var_2.append(url)
                share_lock_2.release()

                upload_flag = 1

                share_lock_3.acquire()
                share_var_3.set(upload_flag)
                share_lock_3.release()

                upload_blog = True

        elif blog == 3:
            if upload_blog:
                upload_blog = False
            else:

                upload_flag = 0

                share_lock_3.acquire()
                share_var_3.set(upload_flag)
                share_lock_3.release()
