"""
External Version 2.1
Internal Version 4.1
2021/05/25 by Joshua Wem from HEU & XiaoLong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

from lib.model import detector_factory

# for video count
import torch
import time
import imageio
from lib.model import detect_and_count

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt, parameters):
    p = parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug,
                    1)
    Detector = detector_factory[
        opt.task]
    detector = Detector(opt)

    if os.path.isdir(opt.demo):
        image_names = []
        video_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
            elif ext in video_ext:
                video_names.append(os.path.join(opt.demo, file_name))

        for (image_name) in image_names:
            ret = detector.run(image_name)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)

        for (video_name) in video_names:
            detector.pause = False

            cap = cv2.VideoCapture(video_name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            video_output_path = os.path.join(opt.demo, os.path.basename(video_name).split('.')[0] + '_res' + '.mp4')
            video_output_writer = imageio.get_writer(video_output_path, fps=p.fps if fps == float('Inf') else fps)

            frame_counter = 0
            counter = 0
            dets = torch.Tensor()
            start_time = time.time()
            while True:
                reta, img = cap.read()
                if reta:
                    start_time_d4c = time.time()
                    img, counter, dets, ret = detect_and_count(opt, img, detector, counter, dets, width, height)
                    b, g, r = cv2.split(img)
                    img_1 = cv2.merge([r, g, b])
                    end_time_d4c = time.time()
                    video_output_writer.append_data(img_1)
                    frame_counter += 1

                    time_str = ''
                    for stat in time_stats:
                        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                    print('\rvideo frame: {} | d4c {:.3f}s |'.format(frame_counter, end_time_d4c - start_time_d4c) + time_str, end='')
                    if cv2.waitKey(1) == 27:
                        video_output_writer.close()
                        print('\nvideo results saved in {} !'.format(video_output_path))
                        break
                else:
                    # video_saved.release()
                    video_output_writer.close()
                    print('\nvideo results saved in {} !'.format(video_output_path))
                    break
            print('All Processes Cost Time: {:.3f}s'.format(time.time() - start_time))

    elif opt.demo == 'webcam':
        detector.pause = False

        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_output_path = os.path.join(opt.video_save_dir, 'webcam_res.mp4')
        video_output_writer = imageio.get_writer(video_output_path, fps=p.fps if fps == float('Inf') else fps)

        frame_counter = 0
        counter = 0
        dets = torch.Tensor()
        start_time = time.time()
        while True:
            reta, img = cap.read()
            if reta:
                start_time_d4c = time.time()
                img, counter, dets, ret = detect_and_count(opt, img, detector, counter, dets, width, height)
                b, g, r = cv2.split(img)
                img_1 = cv2.merge([r, g, b])
                end_time_d4c = time.time()
                video_output_writer.append_data(img_1)
                frame_counter += 1

                time_str = ''
                for stat in time_stats:
                    time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                print('\rvideo frame: {} | d4c {:.3f}s |'.format(frame_counter, end_time_d4c - start_time_d4c) + time_str, end='')
                if cv2.waitKey(1) == 27:
                    video_output_writer.close()
                    print('\nvideo results saved in {} !'.format(video_output_path))
                    break
            else:
                video_output_writer.close()
                print('\nvideo results saved in {} !'.format(video_output_path))
                break
        print('All Processes Cost Time: {:.3f}s'.format(time.time() - start_time))
    else:
        if opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
            detector.pause = False

            cap = cv2.VideoCapture(opt.demo)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            video_output_path = os.path.join(os.path.dirname(opt.demo), os.path.basename(opt.demo).split('.')[0] + '_res' + '.mp4')
            video_output_writer = imageio.get_writer(video_output_path, fps=p.fps if fps == float('Inf') else fps)

            frame_counter = 0
            counter = 0
            dets = torch.Tensor()
            start_time = time.time()
            while True:
                reta, img = cap.read()
                if reta:
                    start_time_d4c = time.time()
                    img, counter, dets, ret = detect_and_count(opt, img, detector, counter, dets, width, height)
                    b, g, r = cv2.split(img)
                    img_1 = cv2.merge([r, g, b])
                    end_time_d4c = time.time()
                    video_output_writer.append_data(img_1)
                    frame_counter += 1

                    time_str = ''
                    for stat in time_stats:
                        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                    print('\rvideo frame: {} | d4c {:.3f}s |'.format(frame_counter,
                                                                     end_time_d4c - start_time_d4c) + time_str, end='')
                    if cv2.waitKey(1) == 27:
                        # video_saved.release()
                        video_output_writer.close()
                        print('\nvideo results saved in {} !'.format(video_output_path))
                        break
                else:
                    video_output_writer.close()
                    print('\nvideo results saved in {} !'.format(video_output_path))
                    break
            print('All Processes Cost Time: {:.3f}s'.format(time.time() - start_time))
        elif opt.demo[opt.demo.rfind('.') + 1:].lower() in image_ext:
            ret = detector.run(opt.demo)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
