import h5py
import sys
import os
from io import BytesIO
from tempfile import TemporaryFile, NamedTemporaryFile

txtname = 'key.txt'


def main():
    equipmentNum = getEquipmentNum()
    user_key = creat_user_key(equipmentNum)
    key = computKeyToDecode(equipmentNum, user_key)

    encrypt('./temp/model_best.pth', './temp/pcia_v7_b', key)  # src_file_path：待加密文件路径  dec_file_path：加密后文件路径
    f = open(txtname, 'a+')
    f.write(equipmentNum + '\n' + user_key + '\n')
    f.close()


def getEquipmentNum():
    import time
    import uuid
    time_now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    equipmentNum = mac + time_now
    equipmentNum = equipmentNum[6:12] + equipmentNum[-7:-3] + equipmentNum[2:4] + \
                   equipmentNum[17:25] + equipmentNum[5:7] + equipmentNum[-3:]

    equipmentNum = equipmentNum.upper()
    output = "-".join([equipmentNum[i:i + 5] for i in range(0, 25, 5)])

    print('EQUIPMENT NUM: ', output)
    return output


def creat_user_key(equipmentNum):
    step_1 = equipmentNum[2:4] + equipmentNum[6:8] + equipmentNum[12:14] + equipmentNum[-4:]
    # print('step1: ', step_1)
    step_2 = step_1[1:4] + step_1[7:] + step_1[0] + step_1[6] + step_1[4:6]
    # print('step2: ', step_2)
    change_num = ord(equipmentNum[-1]) - ord(equipmentNum[-2])

    step_3 = ''
    for temp in step_2:
        step_3 = step_3 + chr(ord(temp) + change_num)
    # print('step3: ', step_3, '  change_num: ', change_num)
    # keyRight = '505505home'
    keyRight = 'JoshuaWen6'
    keyOutput = ''
    for i in range(0, len(keyRight), 2):
        temp_1 = ord(keyRight[i]) - ord(step_3[i])
        temp_2 = ord(keyRight[i + 1]) - ord(step_3[i + 1])
        temp_3 = ''
        if temp_1 < 0 and temp_2 < 0:
            temp_3 = 'F'
        if temp_1 >= 0 and temp_2 < 0:
            temp_3 = 'M'
        if temp_1 < 0 and temp_2 >= 0:
            temp_3 = 'N'
        if temp_1 >= 0 and temp_2 >= 0:
            temp_3 = 'Z'

        temp = '%02d%s%02d' % (abs(temp_1), temp_3, abs(temp_2))
        keyOutput = keyOutput + temp + '-'
    print('KEYOUTPUT NUM: ', keyOutput[:-1])
    return keyOutput[:-1]


def computKeyToDecode(equipmentNum, keyUser):
    import re

    key = ''
    step_1 = equipmentNum[2:4] + equipmentNum[6:8] + equipmentNum[12:14] + equipmentNum[-4:]
    # print('step1: ', step_1)
    step_2 = step_1[1:4] + step_1[7:] + step_1[0] + step_1[6] + step_1[4:6]
    # print('step2: ', step_2)
    change_num = ord(equipmentNum[-1]) - ord(equipmentNum[-2])

    step_3 = ''
    for temp in step_2:
        step_3 = step_3 + chr(ord(temp) + change_num)
    # print('step3: ', step_3, '  change_num: ', change_num)

    changeNumList = []
    key_list = keyUser.split('-')
    for temp in key_list:
        temp_1 = 0
        temp_3 = 0
        matchObj = re.match(r'([0-9]{2})([A-Z]{1})([0-9]{2})', temp)
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
            return "Equipment Error !"

        changeNumList.append(temp_1)
        changeNumList.append(temp_3)

    for i in range(len(changeNumList)):
        key = key + chr(ord(step_3[i]) + changeNumList[i])
    print('key: ', key.encode('ascii'))
    return key.encode('ascii')


def encrypt(inputPath, outputPath, key):
    f_input = open(inputPath, 'rb')

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
    f_output = open(outputPath, 'wb')
    f_output.write(bytes(data_output))
    f_output.close()

    print('encrypt success')


def decode_model(inputPath, outputPath, key):
    f_input = open(inputPath, 'rb')

    data_all = []

    print('compute data')
    for data in f_input.read():
        data ^= key
        data_all.append(data)

    print('write data')
    f_output = open(outputPath, 'wb')
    # f_output = NamedTemporaryFile(dir=outputPath)
    f_output.write(bytes(data_all))

    f_output.close()
    f_input.close()

    print('encrypt success')


def show_h5(filepath):
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    def printname(name):
        print(name)

    f.visit(printname)

    if len(f.attrs.items()):
        print("Root attributes:")
    for key, value in f.attrs.items():
        print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

    for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
        print("  {}".format(layer))

    f.close()


if __name__ == '__main__':
    main()
