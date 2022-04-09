from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import tempfile
import datetime
import torch
from torch.nn import DataParallel

use_cuda = True
use_multi_gpu = True
select_gpu = [0]


def computKeyToDecode(equipmentNum, keyUser):
    import re
    key = ''
    step_1 = equipmentNum[2:4] + equipmentNum[6:8] + equipmentNum[12:14] + equipmentNum[-4:]
    step_2 = step_1[1:4] + step_1[7:] + step_1[0] + step_1[6] + step_1[4:6]
    change_num = ord(equipmentNum[(-1)]) - ord(equipmentNum[(-2)])
    step_3 = ''
    for temp in step_2:
        step_3 = step_3 + chr(ord(temp) + change_num)

    changeNumList = []
    key_list = keyUser.split('-')
    for temp in key_list:
        temp_1 = 0
        temp_3 = 0
        matchObj = re.match('([0-9]{2})([A-Z]{1})([0-9]{2})', temp)
        if matchObj:
            temp_2 = str(matchObj.group(2))
            if temp_2 == 'Z':
                temp_1 = int(matchObj.group(1))
                temp_3 = int(matchObj.group(3))
            if temp_2 == 'F':
                temp_1 = -int(matchObj.group(1))
                temp_3 = -int(matchObj.group(3))
            if temp_2 == 'M':
                temp_1 = int(matchObj.group(1))
                temp_3 = -int(matchObj.group(3))
            if temp_2 == 'N':
                temp_1 = -int(matchObj.group(1))
                temp_3 = int(matchObj.group(3))
        else:
            return 'Equipment Error !'
        changeNumList.append(temp_1)
        changeNumList.append(temp_3)

    for i in range(len(changeNumList)):
        key = key + chr(ord(step_3[i]) + changeNumList[i])

    return key.encode('ascii')


def get_time(flag=0):
    datetimenow = datetime.datetime.now()
    date = datetimenow.date().isoformat()
    time = datetimenow.time().strftime('%H-%M-%S-%f')

    if flag == 0:
        return date + "-" + time
    if flag == 1:
        return date
    if flag == 2:
        return time


class model():

    def __init__(self, equipment_key, usr_key, model_path):
        # key = computKeyToDecode(equipment_key, usr_key)
        key = b'505505home'
        f_input = open(model_path, 'rb')
        print('compute data')
        data_output = []
        key_len = len(key)
        while True:
            data = f_input.read(key_len)
            if not data:
                break
            for i in range(len(data)):
                data_output.append(data[i] ^ key[i])
        f_input.close()
        print('write data')
        f_output = tempfile.NamedTemporaryFile()
        f_output.write(bytes(data_output))

        self.__model = torch.jit.load(f_output.name)

        cuda_available = torch.cuda.is_available()
        num_gpu = torch.cuda.device_count()
        print('use cuda:', cuda_available, '  num gpu:', num_gpu, '  select gpu', select_gpu)

        if cuda_available:
            if len(select_gpu) == 1:
                torch.cuda.set_device(select_gpu[0])
            self.__model = self.__model.cuda()
            use_cuda = True
            if len(select_gpu) > 1 and num_gpu >= len(select_gpu):
                use_multi_gpu = True
                self.__model = DataParallel(self.__model, device_ids=select_gpu)
            else:
                use_multi_gpu = False
        else:
            use_cuda = False
            use_multi_gpu = False

    def get_resules(self, image_list):
        self.__model.eval()
        with torch.no_grad():
            result_list = []

            for idx_, img_Data in enumerate(image_list):

                img_Data = np.asarray((bytearray(img_Data)), dtype='uint8')

                input_string = cv2.imdecode(img_Data, cv2.IMREAD_UNCHANGED)
                input_string = input_string.astype(np.float32)
                input_string = torch.from_numpy(input_string)
                input_string = input_string.unsqueeze(0)
                input_string = input_string.unsqueeze(0)
                input_string = input_string.cuda()

                print('{time}: model running for picture {num}'.format(time=get_time(0), num=idx_))
                predictions = self.__model(input_string)

                if use_cuda:
                    predictions = predictions.cpu().detach().numpy().ravel()
                else:
                    predictions = predictions.detach().numpy().ravel()
                result_list.append(predictions[0])

        return result_list
