# NeRF神经辐射场
import torch
import numpy as np
import configargparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设备使用GPU或CPU
np.random.seed(0) # 生成随机数种子
DEBUG = False # 调试符号
# 参数设置函数
def config_parser():
    parser = configargparse.ArgumentParser() # 设置参数
    # 基本参数
    parser.add_argument('--config',is_config_file=True,help='config file path') # 生成config.txt文件
    parser.add_argument('--expname',type=str,help='experiment name') # 指定项目名称
    parser.add_argument('--basedir',type=str,default='./logs/',help='where to store ckpts and logs') # 指定结果存储目录
    parser.add_argument('--datadir',type=str,default='./data/llff/fern',help='input data directory') # 指定数据集目录
    # 训练参数
    parser.add_argument('--netdepth',type=int,default=8,help='layers in network') # 设置网络深度,即网络层数
    parser.add_argument('--netwidth',type=int,default=256,help='channels per layer') # 设置网络宽度,即每一层神经元个数
    parser.add_argument('--netdepth_fine',type=int,default=8,help='layers in fine network') # 设置细网络深度
    parser.add_argument('--netwidth_fine',type=int,default=256,help='channels per layer in fine network') # 设置细网络宽度
    parser.add_argument('--N_rand',type=int,default=32*32*4,help='batch size (number of random rays per gradient step)') # 设置batch_size,每个梯度步长取样本数,即射线数量
    parser.add_argument('--lrate',type=float,default=5e-4,help='learning rate') # 设置学习率
    parser.add_argument('--lrate_decay',type=int,default=250,help='exponential learning rate decay (in 1000 steps)') # 设置指数学习率衰减,每1000步的衰减
    parser.add_argument('--chunk',type=int,default=1024*32,help='number of rays processed in parallel,decrease if running out of memory') # 设置并行处理的最大ray射线数,防止内存溢出,int类型
    parser.add_argument('--netchunk',type=int,default=1024*64,help='number of pts sent through network in parallel,decrease if running out of memory') # 设置并行发送pt点数量,防止内存溢出
    parser.add_argument('--no_batching',action='store_true',help='only take random rays from 1 image at a time') # 不使用批处理,一次只能从一张图像中获取随机ray射线
    parser.add_argument('--no_reload',action='store_true',help='do not reload weights from saved ckpt') # 不从保存的ckpt模型中加载权重
    parser.add_argument('--ft_path',type=str,default=None,help='specific weights npy file to reload for coarse network') # 为粗网络重新加载特定权重npy文件
    parser.add_argument('--precrop_iters',type=int,default=0,help='number of steps to train on central crops') # 中心crop的训练步数
    parser.add_argument('--precrop_frac',type=float,default=.5,help='fraction of img taken for central crops') # 中心crop的图像分数
    # 渲染参数
    parser.add_argument('--N_samples',type=int,default=64,help='number of coarse samples per ray') # 每条射线的粗采样点数
    parser.add_argument('--N_importance',type=int,default=0,help='number of additional fine samples per ray') # 每条射线附加的细采样点数
    parser.add_argument('--perturb',type=float,default=1.0,help='set to 0. for no jitter,1. for jitter') # 是否抖动
    parser.add_argument('--use_viewdirs',action='store_true',help='use full 5D input instead of 3D') # 是否使用5D坐标取代3D坐标
    parser.add_argument('--i_embed',type=int,default=0,help='set 0 for default positional encoding,-1 for none') # 是否使用位置编码
    parser.add_argument('--multires',type=int,default=10,help='log2 of max freq for positional encoding (3D location)') # 3D位置的位置编码的最大频率
    parser.add_argument('--multires_views',type=int,default=4,help='log2 of max freq for positional encoding (2D direction)') # 2D方向的位置编码的最大频率
    parser.add_argument('--raw_noise_std',type=float,default=0.0,help='std dev of noise added to regularize sigma_a output,1e0 recommended') # 是否加入噪音方差
    parser.add_argument('--render_only',action='store_true',help='do not optimize,reload weights and render out render_poses path') # 不优化,重新加载权重渲染render_poses
    parser.add_argument('--render_test',action='store_true',help='render the test set instead of render_poses path') # 渲染测试数据而不是render_poses路径
    parser.add_argument('--render_factor',type=int,default=0,help='downsampling factor to speed up rendering,set 4 or 8 for fast preview') # 设置下采样因子以加快渲染速度
    # 数据集参数
    parser.add_argument('--dataset_type',type=str,default='llff',help='options: llff / blender / deepvoxels') # 数据集选择:llff或blender或deepvoxels
    parser.add_argument('--testskip',type=int,default=8,help='will load 1/N images from test/val sets,useful for large datasets like deepvoxels') # 从deepvoxels数据集中加载1/N图像
    parser.add_argument('--shape',type=str,default='greek',help='options : armchair / cube / greek / vase') # 数据集deepvoxels选项
    parser.add_argument('--white_bkgd',action='store_true',help='set to render synthetic data on a white bkgd (always use for dvoxels)') # 数据集blender选项,是否在白色bkgd上渲染
    parser.add_argument('--half_res',action='store_true',help='load blender synthetic data at 400x400 instead of 800x800') # 数据集blender格式
    parser.add_argument('--factor',type=int,default=8,help='downsample factor for LLFF images') # 数据集llff的下采样因子
    parser.add_argument('--no_ndc',action='store_true',help='do not use normalized device coordinates (set for non-forward facing scenes)') # 不使用标准化坐标(为非正面场景设置)
    parser.add_argument('--lindisp',action='store_true',help='sampling linearly in disparity rather than depth') # 在视差而不是深度上线性采样
    parser.add_argument('--spherify',action='store_true',help='set for spherical 360 scenes') # 设置为球面360度场景
    parser.add_argument('--llffhold',type=int,default=8,help='will take every 1/N images as LLFF test set,paper uses 8') # 从llff数据集中加载1/N图像
    # 存储参数
    parser.add_argument('--i_print',type=int,default=100,help='frequency of console printout and metric loggin') # 控制台打印输出和日志的频率
    parser.add_argument('--i_img',type=int,default=500,help='frequency of tensorboard image logging') # 可视化tensorboard绘制图像频率
    parser.add_argument('--i_weights',type=int,default=10000,help='frequency of weight ckpt saving') # 权重ckpt=checkpoint保存频率
    parser.add_argument('--i_testset',type=int,default=50000,help='frequency of testset saving') # 测试数据集保存频率
    parser.add_argument('--i_video',type=int,default=50000,help='frequency of render_poses video saving') # 渲染视频保存频率
    return parser
# 模型训练函数
def train():
    print('~~~模型训练函数开始~~~')
    parser = config_parser() # 设置参数
    args = parser.parse_args() # 读取参数
    K = None
    print('>>>第一步,加载数据集')
    print('>>>第二步,加载数据集')
    print('>>>第三步,加载数据集')
    print('>>>第四步,加载数据集')
    print('~~~模型训练函数结束~~~')
# 主函数
if __name__=='__main__':
    print('~~~主函数开始~~~')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
    print('~~~主函数结束~~~')