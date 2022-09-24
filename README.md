# stereo_vision
three-dimensional reconstruction in machine vision
## 环境配置
Windows安装miniconda，添加环境变量
```
C:\ProgramData\Miniconda3\condabin
C:\ProgramData\Miniconda3\Scripts
C:\ProgramData\Miniconda3\Library\bin
```
Ubuntu安装miniconda
```
sudo apt-get install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
反复回车
yes
~/.miniconda3
yes
source ~/.bashrc
conda --version
```
Conda安装各种软件包
```
conda create -n torch python=3.9
activate torch
conda info --envs
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge imageio
conda install -c conda-forge imageio-ffmpeg
conda install -c conda-forge matplotlib
conda install -c conda-forge configargparse
conda install -c conda-forge tensorboardx
conda install -c conda-forge tqdm
conda install -c conda-forge opencv
```
下载数据集，运行训练代码
```
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
activate torch
python run_nerf.py --config configs/wyf.txt
```