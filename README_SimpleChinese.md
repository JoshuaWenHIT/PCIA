# <div align="center">PCIA</div>
<div align="center">由哈尔滨工程大学智能信息处理实验室和北京小龙潜行科技有限公司合作的项目</div>

## <div align="center">什么是PCIA ?</div>

![](data/images/graph_abstract.png "graph abstract")

**PCIA 是 Pig Counting In Aisles 的缩写。**

这个项目旨在固定视角视频场景下对通道中的运动目标进行连续计数。并且，可指定运动正方向，当运动目标反方向通过时则减量计数。

## <div align="center">如何安装PCIA ?</div>

PCIA分为在线和本地两个版本。由于现场网络的不稳定性，通常只会部署本地版本以避免由于网络异常带来的云上传与推流中断。

<details open>
<summary>本地版安装-准备工作</summary>

**步骤 0: 系统检查（通常这一步公司的工作人员会替你完成，所以你可以直接跳转到步骤1）**

首先，检查 Ubuntu 系统是否安装在固态硬盘中，若安装在机械盘中请重新安装系统。 之后，关闭系统 BIOS 中的安全启动项。

解决重启后向日葵连接不上的问题：

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install lightdm
```

**步骤 1: NVIDIA显卡驱动安装**

更换系统源为阿里云源： 点击Software & Updates图标， 在界面内将Download from项选择为http://mirrors.aliyun.com/ubuntu, 点击Close，点击Reload。

再次点击Software & Updates图标，选择Additional Drivers，点击下载450、460、470版本NVIDIA显卡驱动，安装成功后，重启主机。
重启后，点击Setting图标，查看Graphics项是否变为NVIDIA驱动，并在终端输入：

```bash
nvidia-smi
```

查看是否可查看显卡占用信息。如果均满足，则显卡驱动安装成功。

**步骤 2: Anaconda安装**

从Anaconda官网下载最新安装包，终端运行命令：

```bash
bash {Anconda安装包名}.sh
```

全程选择yes，重启终端，如果行头出现（base）表示安装成功。

**步骤 3: Pycharm安装**

从JetBrain官网下载最新版本，将其拷贝至home所在目录，终端运行命令解压：

```bash
sudo tar zxvf {Pycharm安装包名}.tar.gz
cd {解压后文件夹名}/bin
sh ./pycharm.sh
```

**记得使用自己的学生账号登录pycharm将产品激活。**

**步骤 4: CUDA和cuDNN安装（请保证步骤1完成妥当）**

从NVIDIA官网CUDA archive中下载CUDA-11.0版本，按照官方要求进行安装操作。

从NVIDIA官网下载对应CUDA-11.0的cuDNN最新版本，终端输入：

```bash
tar zxvf {cuDNN压缩包名}.tar.gz
sudo cp cuda/include/cudnn* /usr/local/cuda-11.0/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.0/lib64/
sudo chmod a+r /usr/local/cuda-11.0/include/cudnn*
sudo chmod a+r /usr/local/cuda-11.0/lib64/libcudnn*
```

添加环境变量，终端输入：

```bash
sudo gedit ~/.bashrc
```
在文档空白处添加文本：

```bash
# CUDA config
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64
export PATH=$PATH:/usr/local/cuda-11.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.0
```

**步骤 5: Git安装**

终端输入：

```bash
sudo apt install git
```

**步骤 6: conda换源**

终端输入：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

**步骤 7: pip换源**

在home目录下创建.pip文件夹，在文件夹中创建pip.conf文件，内容如下：

```bash
[global]
timeout = 6000
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

</details>

<details open>
<summary>本地版安装-源码安装</summary>

**步骤 1: 源码clone**

可以直接从本repo下的code文件夹中复制PCIA-local文件夹，作为源码文件夹。

或者从我的其他repo中clone：

```bash
git clone https://github.com/JoshuaWenHIT/PCIA-local.git
```

**步骤 2: 环境配置**

相关库安装：

```bash
pip install -r requirements.txt
```

深度学习框架安装（建议采用我指定的版本）：

```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 cudnn=7.6.5
```

推流库安装（建议采用我指定的版本）：

```bash
conda install ffmpeg==4.3.1
```

**步骤 3: DCNv2编译（可省略，demo.py运行出错时尝试运行该步骤）**

终端输入：

```bash
cd DCNv2
export CUDA_HOME=/usr/local/cuda-11.0
python setup.py build develop
```

如果前面的步骤操作没有问题，这一步自然会成功。如果失败，请从头检查各步骤是否存在不妥之处。

</details>

## <div align="center">如何使用PCIA ?</div>

在完成安装步骤之后，就可以运行代码了。
首先，需要对部署好的代码进行测试，这部分测试主要分为两部分：demo测试和main测试。

<details open>
<summary>demo测试</summary>



</details>