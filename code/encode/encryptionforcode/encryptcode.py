# coding:utf-8
"""
功能：python代码加密程序，使用Cython的方法对代码进行加密，加密后生成so文件
Terminal: python encryptcode.py build_ext --inplace
2020/01/17 by Nitin from HEU
"""
from distutils.core import setup
from Cython.Build import cythonize
import os

num_flag = False
code_path = r"process/"


if num_flag:
    key_funs = ['workstation.py']  # 需要加密的py文件列表
else:
    for root, sub, file in os.walk(code_path):
        key_funs = []
        if len(file) == 0:
            continue
        # for idx, filename in enumerate(key_funs):
        #     if filename.__contains__(".py"):
        #         process_list.append(os.path.join(root, filename))
        os.chdir(root)
        for idx, filename in enumerate(file):
            if filename.__contains__(".py"):
                key_funs.append(filename)

        setup(
            name="Code Encryption for Python",
            ext_modules=cythonize(key_funs),
        )

        files = os.listdir(os.getcwd())
        # 删除生成的中间文件和源文件
        for fi in files:
            if fi.__contains__(".pyd"):
                re_name = fi.split(".")[0] + ".pyd"
                print(re_name)
                os.rename(fi, re_name)
            elif fi.__contains__(".c") or fi in key_funs:
                print(fi)
                if fi.__contains__(".cpython"):
                    pass
                else:
                    os.remove(fi)

        # 删除build，其实有个包可以一步完成
        for root, dirs, files in os.walk('build', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir('build')

print('Code encrypt done!')

