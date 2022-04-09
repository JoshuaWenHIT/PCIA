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

或者从我的其他repo中clone（暂未开放clone）：

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

## <div align="center">如何测试PCIA ?</div>

在完成安装步骤之后，需要对部署代码进行初步测试。
这部分测试主要分为3个部分：demo测试、main测试和upload测试。

<details open>
<summary>demo测试</summary>

demo.py文件默认使用test2.mp4作为测试视频，可直接运行：

```bash
python demo.py
```

检测结果会在./video文件夹下生成test2_res.mp4结果视频，如果整个计算过程正常且结果视频正常，则demo测试通过。

值得一提的是，demo测试主要用于测试检测计数网络部署是否成功，并不涉及PCIA产品部分的通信、推流和云上传等任务。
这部分的测试则主要在main测试中完成。


</details>

<details open>
<summary>main测试</summary>

main测试可以分为3个步骤进行：1）代码授权； 2）代码启动； 3）联调。

**步骤 1: 代码授权**

这部分涉及./lib/model.py中的代码加密部分，加密部分的内容会在“如何加密PCIA？”部分详细说明，这里仅对运行所使用的license.lic文件的生成做出说明。

在./code文件夹中，encode文件夹表示加密功能文件。生成license.lic文件前，需要知道部署主机的mac地址：

```bash
ifconfig
```

<div align="center">
<img src="data/images/get_mac.png">
</div>

以上图为例，分别获得了3组参数，其中enp0s31f6为正在使用的网卡型号，该型号会根据不同设备而不同。

**请注意自己的设备网卡型号！！！并在PCIA-local文件夹中./lib/model.py的line87进行修改！！！**

mac地址在ether后表示： b0:6e:bf:60:2a:24，获取mac地址后在终端输入（在encode根目录下）：

```bash
python CreateLicense.py $your_macAddress-$month-$day # e.g. b06ebf602a24-05-01
```

其中，$your_macAddress填充所部署设备的mac地址，$month-$day填写license.lic文件授权截止日期（如果想永久授权可以填写13-01，即正常月份中不存在的月份）。
运行后可在encode根目录下获得license.lic文件，将该文件复制到PCIA-local根目录下，即可完成授权。

**步骤 2: 代码启动**

完成授权后，在main.py中进行修改：1）修改line18的mac地址；2）修改line20的摄像头地址。完成后在终端输入：

```bash
python main.py
```

如果运行正常，可观察到如下输出：

```bash
已授权!
生成模型...
加载模型 ./weights/pcia_v7_b

模型加载完毕!
等待接收服务器指令......
```

**步骤 3: 联调**

需要联系公司人员进行配合测试，公司工作人员会使用平板控制算法的开启和结束，需要实际操作演练。

</details>

<details open>
<summary>upload测试</summary>

**需保证./video/output下存有合法命名视频，建议在main测试步骤3联调后进行！！！**

终端输入：

```bash
python upload.py
```

如果终端显示上传成功，则表示upload测试通过。

</details>

## <div align="center">如何加密PCIA ?</div>

在完成测试工作后，出于技术保密角度，需要对库文件代码和网络权重文件进行加密。

在“如何测试PCIA？”中，说明了代码授权方法，下面将从剩余2个方面进行说明：1）代码加密；2）模型加密。

**步骤 1: 代码加密**

代码加密主要将.py文件通过Cython方法加密为.so库文件，放在终端进行运行。在encode/encryptionforcode下将待加密的.py文件复制到process文件夹，终端运行：

```bash
python encryptcode.py build_ext --inplace
```

可在process文件夹下获得加密好的.so文件将其一一对应替换掉PCIA-local的./lib中的.py文件，即可完成代码加密。

**步骤 2: 模型加密**

模型加密主要将.pth文件通过密码本哈希校验的方法加密为无后缀名的权重文件，在PCIA-local中./weights文件夹下已经存在已加密的文件。
将待加密的.pth文件复制到encode/encryptionformodel/encryption/temp文件夹，
修改encode/encryptionformodel/encryption/main.py中的line15，自定义输入权重文件名和输出文件路径，
最后终端运行：

```bash
python main.py # 注意此处的main.py文件路径为encode/encryptionformodel/encryption/main.py，请勿与PCIA-local下的main.py弄混
```

可在temp文件夹下获得加密好的模型权重文件，将其替换掉PCIA-local的./weights中的.pth文件，即可完成模型加密。

**建议加密后重新进行上述测试，以保证加密无误！！！**