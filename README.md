# <div align="center">PCIA</div>
<div align="center">The cooperation project of Harbin Engineering University and Beijing Focused Loong Technology Company Limited.</div>

## <div align="center">What is PCIA ?</div>

![](data/images/graph_abstract.png "graph abstract")

**PCIA is an acronym for Pig Counting In Aisles.**

The purpose of this project is to continuously count the moving objects in aisles in a fixed-view video scene, 
and specify the positive direction of movement, 
which means that the objects moving in the opposite direction should be counted down.

## <div align="center">How to install PCIA ?</div>

PCIA is divided into an online version and a local version. 
Due to the instability of the on-site network, 
only the local version is usually deployed on the device to prevent interruption of cloud uploading and streaming caused by network abnormalities.

<details open>
<summary>Local Version Install</summary>

**Step 0: System Check. (Usually the staff of the company will do it for you, so you can jump to Step 1.)**

First, check whether the Ubuntu system is installed in the solid state drive, if installed in the mechanical disk, please reinstall the system.
After that, turn off the Secure Boot entry in the system BIOS.

Solve the problem that Sunflower cannot connect after restart:
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install lightdm
```

**Step 1: NVIDIA Graph Device Driver Install**

Change the system source to Aliyun source: Click the Software & Updates icon, select the Download from item in the interface as http://mirrors.aliyun.com/ubuntu, click Close, and click Reload.

Click the Software & Updates icon again, select Additional Drivers, and click to download the 450, 460, and 470 NVIDIA graphics card drivers. After the installation is successful, restart the host.
After restarting, click the Setting icon to see if the Graphics item has changed to the NVIDIA driver, and type in the terminal:

```bash
nvidia-smi
```

Check whether the graphics card usage information can be viewed. If all are satisfied, the graphics card driver is installed successfully.

**Step 2: Anaconda Install**

Download the latest installation package from the Anaconda official website and run the command in the terminal:

```bash
bash {Anconda Package Name}.sh
```

Select yes in the whole process, restart the terminal, if the line header appears (base), the installation is successful.

**Step 3: Pycharm Install**

Download the latest version from the JetBrain official website, copy it to the directory where your home is located, and run the command in the terminal to decompress it:

```bash
sudo tar zxvf {Pycharm Package Name}.tar.gz
cd {Unzipped dictory name}/bin
sh ./pycharm.sh
```

**Remember to log in to pycharm with your student account to activate the product.**

**Step 4: CUDA and cuDNN Install（Please ensure that Step 1 is completed properly.）**

Download the CUDA-11.0 version from the CUDA archive on the NVIDIA official website and install it according to the official requirements.

Download the latest version of cuDNN corresponding to CUDA-11.0 from the NVIDIA official website, and enter:

```bash
tar zxvf {cuDNN Package Name}.tar.gz
sudo cp cuda/include/cudnn* /usr/local/cuda-11.0/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.0/lib64/
sudo chmod a+r /usr/local/cuda-11.0/include/cudnn*
sudo chmod a+r /usr/local/cuda-11.0/lib64/libcudnn*
```

Add environment variables, terminal input:

```bash
sudo gedit ~/.bashrc
```
Add text in the blanks of the document:

```bash
# CUDA config
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64
export PATH=$PATH:/usr/local/cuda-11.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.0
```

**Step 5: Git Install**

Terminal input：

```bash
sudo apt install git
```

**步骤 6: Replace the source of conda**

Terminal input：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

**Step 7: Replace the source of conda**

Create a .pip folder in the home directory, and create a pip.conf file in the folder with the following contents:

```bash
[global]
timeout = 6000
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

</details>