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



Clone repo and install [requirements.txt](code/PCIA-local/requirements.txt)

</details>