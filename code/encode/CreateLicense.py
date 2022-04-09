# -*- coding: UTF-8 -*-
"""
功能：根据MAC地址生成lic文件，需要事先知道MAC地址
Terminal: python CreateLicense.py $your macAddress$-$month$-$day$ e.g. b06ebf602a24-05-01
2020/01/17 by Nitin from HEU
2020/04/25 modified by Joshua Wen from HEU
"""
import sys
from Crypto.Cipher import AES
import binascii

# 秘钥，编解码要写一样的
# seperateKey = "505505@home"
# aesKey = "11223344aabbccdd"  # 密钥
# aesIv = "55667788eeffgghh"    # 偏移量
seperateKey = "JoshuaWen66"
aesKey = "jojojojo12345678"  # 密钥
aesIv = "jo19950315dragon"    # 偏移量
aesMode = AES.MODE_CBC  # 四种模式，ECB、CBC(密码分组链接模式)、CFB、OFB


def encrypt(text):
    crypt = AES.new(aesKey.encode('utf-8'), aesMode, aesIv.encode('utf-8'))   # 创建aes对象
    # 要求长度是16，所以要padding，对应的解码程序没写完善
    add, length = 0, 16
    count = len(text)
    if count % length != 0:
        add = length - (count % length)
    text = text + ('6' * add)   # 6是自己定的，随便,但不要与MAC地址最后一位重复
    ciphertext = crypt.encrypt(text.encode('utf-8'))
    return binascii.hexlify(ciphertext)


if __name__ == "__main__":
    argLen = len(sys.argv)

    if argLen != 2: 
        print("usage: python {} hostInfo".format(sys.argv[0]))
        sys.exit(0)
    
    hostInfo = sys.argv[1]

    # 加密用两层，这个自己定的，但层数多不代表安全
    encryptText = encrypt(hostInfo)     # 第一层加密返回十六进制加密后的数
    encryptText = encryptText.decode() + seperateKey + "Valid"
    encryptText = encrypt(encryptText)  # 第二层加密，返回也是十六进制数
    
    with open("./license.lic", "wb") as licFile:
        licFile.write(encryptText)
        licFile.close()
    
    print("生成license成功!")
