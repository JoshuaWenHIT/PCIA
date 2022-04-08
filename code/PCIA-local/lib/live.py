"""
External Version 1.2
Internal Version 3.2
2021/04/30 by Joshua Wem from HEU & XiaoLong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import copy
import time
import torch
import cv2
import imageio
import subprocess as sp

from lib.upload import get_latest_file
from lib.model import model_builder, detect_and_count


time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def streaming(share_var, share_var_1, share_var_4, share_var_5, share_var_6, share_var_8, share_lock, share_lock_1, share_lock_4, share_lock_6, share_lock_8, parameters, opt):
    p = parameters
    stream_blog = False
    stream_pip_blog = False
    count = 0
    counter = 0
    dets = torch.Tensor()
    message_2 = {}
    output_store_path = ''
    original_store_path = ''

    detector = model_builder(opt)
    print('\n模型加载完毕!')
    print('等待接收服务器指令......')
    auto_start_time = time.time()

    while True:
        share_lock.acquire()
        blog = share_var.get()
        share_lock.release()
        if blog == 2:
            share_lock_6.acquire()
            p.camera_path = 'rtsp://admin:mcdz1234@' + share_var_6["cameraIP"] + '/Streaming/Channels/101'
            # p.camera_path = 'rtmp://' + share_var_6["cameraIP"] + ':1935/myapp/e45f0102e603'
            opt.direction = share_var_6["direction"]
            if share_var_6["autoSave"] == "1":
                p.auto_save = True
            else:
                p.auto_save = False
            if share_var_6["live"] == "1":
                p.live = True
            else:
                p.live = False
            share_lock_6.release()
            share_lock_8.acquire()
            share_var_8["blog_ck1"] = "done"
            share_lock_8.release()
        if blog == 1:
            if not stream_blog:
                cap = cv2.VideoCapture(p.camera_path)
                if p.auto_save:
                    try:
                        auto_count = int(
                            get_latest_file(p.original_auto_path).split('/')[-1].split('.')[0].split('_')[
                                -1]) + 1
                    except:
                        auto_count = 0
                print('\n############################## 开启视频盘点 ##############################')
            if not stream_pip_blog:
                share_lock_6.acquire()
                message_1 = share_var_6["uuid"]
                share_lock_6.release()
                time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime(time.time()))
                print(time_stamp)
                output_store_path = os.path.join(p.output_temp_path + '/' + 'NU_' + time_stamp + message_1 + '.mp4')
                original_store_path = os.path.join(p.original_temp_path + '/' + 'NU_' + time_stamp + message_1 + '.mp4')
                if p.auto_save:
                    output_store_path_auto = os.path.join(p.output_auto_path,
                                                          output_store_path.split('/')[-1].split('.')[0] + '_{}'.format(
                                                              auto_count) + '.mp4')
                    original_store_path_auto = os.path.join(p.original_auto_path,
                                                            original_store_path.split('/')[-1].split('.')[
                                                                0] + '_{}'.format(auto_count) + '.mp4')
                    output_video_saved_auto = imageio.get_writer(output_store_path_auto, fps=p.fps)
                    original_video_saved_auto = imageio.get_writer(original_store_path_auto, fps=p.fps)
                output_video_saved = imageio.get_writer(output_store_path, fps=p.fps)
                original_video_saved = imageio.get_writer(original_store_path, fps=p.fps)
                stream_pip_blog = True
            if cap.isOpened():
                start_time = time.time()
                stream_blog = True
                ret, frame0_rgb = cap.read()
                read_time_flag = time.time()
                if ret:
                    # b, g, r = cv2.split(frame0)
                    # frame0_rgb = cv2.merge([r, g, b])
                    pre_time_flag = time.time()
                    original_video_saved.append_data(frame0_rgb)
                    if p.auto_save:
                        original_video_saved_auto.append_data(frame0_rgb)
                    savein_time_flag = time.time()
                    frame0_rgb, counter, dets, ret = detect_and_count(opt, frame0_rgb, detector, counter, dets, p.width, p.height)
                    saveout_time_flag_1 = time.time()
                    output_video_saved.append_data(frame0_rgb)
                    if p.auto_save:
                        output_video_saved_auto.append_data(frame0_rgb)
                    if p.live:
                        share_var_5.put(frame0_rgb)
                    saveout_time_flag_2 = time.time()
                    live_time_flag = time.time()
                    time_str = ''
                    for stat in time_stats:
                        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                    if p.auto_save:
                        if int(time.time() - auto_start_time) >= p.auto_save_time:
                            auto_count += 1
                            auto_start_time = time.time()
                            try:
                                original_video_saved_auto.close()
                                output_video_saved_auto.close()
                            except:
                                pass
                            try:
                                original_video_saved_auto = imageio.get_writer(os.path.join(p.original_auto_path , original_store_path.split('/')[-1].split('.')[0] + '_{}'.format(auto_count) + '.mp4'), fps=p.fps)
                                output_video_saved_auto = imageio.get_writer(os.path.join(p.output_auto_path , output_store_path.split('/')[-1].split('.')[0] + '_{}'.format(auto_count) + '.mp4'), fps=p.fps)
                            except:
                                pass
                            print("\nauto saved !")
                else:
                    print('\n获取视频帧失败 !')
                    cap.release()
                    cap = cv2.VideoCapture(p.camera_path)
                    continue
                count += 1
                torch.cuda.synchronize()
                end_time = time.time()
                read_time = read_time_flag - start_time
                pre_time = pre_time_flag - read_time_flag
                savein_time = savein_time_flag - pre_time_flag
                saveout_time = saveout_time_flag_2 - saveout_time_flag_1
                live_time = live_time_flag - saveout_time_flag_2
                total_time = end_time - start_time
                print('\rvideo frame: %d |d4c: %.3fs |read: %.3fs |pre: %.3fs |savein: %.3fs |saveout: %.3fs |live: %.3fs |%s' % (count, total_time, read_time, pre_time, savein_time, saveout_time, live_time, time_str), end='')
                time.sleep(0.0001)
            else:
                print('摄像头rtsp连接失败 !')
                cap.release()
                cap = cv2.VideoCapture(p.camera_path)
                time.sleep(0.0001)
        elif blog == 0:
            if stream_blog:
                original_video_saved.close()
                if p.auto_save:
                    original_video_saved_auto.close()
                print('\n原视频已存储 !')
                output_video_saved.close()
                if p.auto_save:
                    output_video_saved_auto.close()
                print('结果视频已存储 !')
                message_2["total_number"] = str(counter)
                shutil.move(original_store_path,
                            os.path.join(p.original_store_path, original_store_path.split('/')[-1]))
                shutil.move(output_store_path,
                            os.path.join(p.output_store_path, output_store_path.split('/')[-1]))
                share_lock_4.acquire()
                share_var_4["resultNumber"] = str(counter)
                share_var_4["inputVideoPath"] = os.path.abspath(os.path.join(p.original_store_path, original_store_path.split('/')[-1]))
                share_var_4["outputVideoPath"] = os.path.abspath(os.path.join(p.output_store_path, output_store_path.split('/')[-1]))
                share_lock_4.release()
                count = 0
                counter = 0
                time.sleep(0.0001)
                stream_blog = False
            if stream_pip_blog:
                cap.release()
                time.sleep(0.0001)
                stream_pip_blog = False
            time.sleep(0.0001)
