"""
External Version 1.0
Internal Version 1.0
2021/04/30 by Joshua Wem from HEU & XiaoLong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess as sp


def streaming_plug(share_var, share_lock, share_var_5, share_var_6, share_lock_6, share_var_8, share_lock_8, parameters):
    p = parameters
    command = []

    count = 0
    live_blog = False
    while True:
        share_lock.acquire()
        blog = share_var.get()
        share_lock.release()
        if blog == 2:
            share_lock_6.acquire()
            # p.rtmpUrl = 'rtmp://' + share_var_6["rtmpIP"] + ':1935/live/' + p.macAddress
            p.rtmpUrl = share_var_6["rtmpIP"] + ':6666'
            share_lock_6.release()
            command = ['ffmpeg',
                       '-loglevel', 'quiet',
                       '-y',
                       # '-f', 'flv',
                       '-f', 'rawvideo',
                       '-vcodec', 'rawvideo',
                       '-pix_fmt', 'bgr24',
                       '-s', "{}x{}".format(p.width, p.height),
                       '-r', str(p.fps),
                       '-i', '-',
                       '-c:v', 'libx264',
                       '-b:v', '500k',
                       '-bufsize', '500k',
                       '-pix_fmt', 'yuv420p',
                       '-preset', 'ultrafast',
                       '-f', 'mpegts',
                       '-flvflags', 'no_duration_filesize',
                       'udp://{}'.format(p.rtmpUrl)]
                       # p.rtmpUrl]

            share_lock_8.acquire()
            share_var_8["blog_ck2"] = "done"
            share_lock_8.release()
        if blog == 1:
            if not live_blog:
                live_blog = True
                print(command)
                push = sp.Popen(command, stdin=sp.PIPE)
                print("\n############################# 开始推流 ##############################")
            frame = share_var_5.get()
            if frame.size != 0:
                try:
                    push.stdin.write(frame.tostring())
                    count += 1
                    print('\rlive frame: %d' % count, end='')
                except:
                    pass
            # print('\rlive frame: %d' % count, end='')
        elif blog == 0:
            # frame = share_var_5.get()
            # if frame.size != 0:
            #     push.stdin.write(frame.tostring())
            if live_blog:
                push.terminate()
                count = 0
                live_blog = False
                print("\n############################# 推流关闭 ##############################")
