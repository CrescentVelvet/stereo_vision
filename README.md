# stereo_vision
three-dimensional reconstruction in machine vision
## 环境配置
安装miniconda，安装各种软件包
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
http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
activate torch
python run_nerf.py --config configs/wyf.txt
```