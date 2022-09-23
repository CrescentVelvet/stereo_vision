# NeRF神经辐射场
import os
import torch
import torch.nn as nn
import numpy as np
import configargparse
import load_data
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
# 嵌入模型类
class Embedder:
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0.,max_freq,steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0.,2.**max_freq,steps=N_freqs)            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x,p_fn=p_fn,freq=freq : p_fn(x * freq))
                out_dim += d                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim        
    def embed(self,inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns],-1)
# NeRF模型类
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False): # 是否输入射线方向
        super(NeRF,self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs # 是否输入射线方向
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch,W)] + [nn.Linear(W,W) if i not in self.skips else nn.Linear(W + input_ch,W) for i in range(D-1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W,W//2)]) # 根据官方代码(https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W,W//2)] + [nn.Linear(W//2,W//2) for i in range(D//2)]) # 根据论文代码
        if use_viewdirs: # 如果输入射线方向
            self.feature_linear = nn.Linear(W,W)
            self.alpha_linear = nn.Linear(W,1)
            self.rgb_linear = nn.Linear(W//2,3)
        else:
            self.output_linear = nn.Linear(W,output_ch)
    def forward(self,x):
        input_pts,input_views = torch.split(x,[self.input_ch,self.input_ch_views],dim=-1)
        h = input_pts
        for i,l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts,h],-1)
        if self.use_viewdirs: # 如果输入射线方向
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature,input_views],-1)        
            for i,l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb,alpha],-1)
        else:
            outputs = self.output_linear(h)
        return outputs
    def load_weights_from_keras(self,weights):
        assert self.use_viewdirs,"Not implemented if use_viewdirs=False"
        for i in range(self.D): # 加载pts_linears
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        idx_feature_linear = 2 * self.D # 加载feature_linear
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))
        idx_views_linears = 2 * self.D + 2 # 加载views_linears
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))
        idx_rbg_linear = 2 * self.D + 4 # 加载rgb_linear
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))
        idx_alpha_linear = 2 * self.D + 6 # 加载alpha_linear
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
# 嵌入函数
def get_embedder(multires,i=0):
    if i == -1:
        return nn.Identity(),3    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin,torch.cos],
    }    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x,eo=embedder_obj : eo.embed(x)
    return embed,embedder_obj.out_dim
# 模型建立函数
def create_nerf(args):
    embed_fn,input_ch = get_embedder(args.multires,args.i_embed) # 调用get_embedder获得一个对应的embedding嵌入函数
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs: # 如果使用5D坐标取代3D坐标
        embeddirs_fn,input_ch_views = get_embedder(args.multires_views,args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4 # 如果每条射线附加的细采样点数大于0则输出5,否则输出4
    skips = [4]
    model = NeRF(D=args.netdepth,# 建立粗网络模型
                 W=args.netwidth,
                 input_ch=input_ch,
                 output_ch=output_ch,
                 skips=skips,
                 input_ch_views=input_ch_views,
                 use_viewdirs=args.use_viewdirs).to(device) # 是否使用5D坐标取代3D坐标
    grad_vars = list(model.parameters()) # 提取粗网络模型梯度
    model_fine = None
    if args.N_importance > 0: # 如果每条射线附加的细采样点数大于0
        model_fine = NeRF(D=args.netdepth_fine,# 建立细网络模型
                          W=args.netwidth_fine,
                          input_ch=input_ch,
                          output_ch=output_ch,
                          skips=skips,
                          input_ch_views=input_ch_views,
                          use_viewdirs=args.use_viewdirs).to(device) # 是否使用5D坐标取代3D坐标
        grad_vars += list(model_fine.parameters()) # 累加细网络模型梯度
    network_query_fn = lambda inputs,viewdirs,network_fn : run_network(inputs,
                                                                       viewdirs,
                                                                       network_fn,
                                                                       embed_fn=embed_fn,
                                                                       embeddirs_fn=embeddirs_fn,
                                                                       netchunk=args.netchunk)
    optimizer = torch.optim.Adam(params=grad_vars,lr=args.lrate,betas=(0.9,0.999)) # 创建Adam优化器
    start = 0
    basedir = args.basedir # 存储目录
    expname = args.expname # 项目名称
    if args.ft_path is not None and args.ft_path!='None': # 为粗网络加载权重checkpoints
        ckpts = [args.ft_path]
    else: # 读取已保存权重checkpoints
        ckpts = [os.path.join(basedir,expname,f) for f in sorted(os.listdir(os.path.join(basedir,expname))) if 'tar' in f]
    print('Found ckpts',ckpts)
    if len(ckpts) > 0 and not args.no_reload: # 如果要从保存的模型中加载权重
        ckpt_path = ckpts[-1] # 把checkpoints数组[-1]作为路径
        print('Reloading from',ckpt_path)
        ckpt = torch.load(ckpt_path) # 读取checkpoints权重
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict']) # 优化器载入权重
        model.load_state_dict(ckpt['network_fn_state_dict']) # 粗网络模型载入权重
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict']) # 细网络模型载入权重
    render_kwargs_train = { # 加载模型
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,# 是否抖动
        'N_importance' : args.N_importance,# 每条射线附加的细采样点数
        'network_fine' : model_fine,# 细网络模型
        'N_samples' : args.N_samples,# 每条射线的粗采样点数
        'network_fn' : model,# 粗网络模型
        'use_viewdirs' : args.use_viewdirs,# 是否使用5D坐标取代3D坐标
        'white_bkgd' : args.white_bkgd,# 数据集blender选项,是否在白色bkgd上渲染
        'raw_noise_std' : args.raw_noise_std,# 是否加入噪音方差
    }
    if args.dataset_type != 'llff' or args.no_ndc: # ndc仅适用于llff类型的前向数据
        print('Not ndc!')
        render_kwargs_train['ndc'] = False # 不使用ndc
        render_kwargs_train['lindisp'] = args.lindisp
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.0
    return render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer
# 模型训练函数
def train():
    print('~~~模型训练函数开始~~~')
    parser = config_parser() # 设置参数
    args = parser.parse_args() # 读取参数
    K = None
    print('>>>第一步,加载数据集',args.dataset_type)
    if args.dataset_type == 'blender': # 加载数据集blender
        images,poses,render_poses,hwf,i_split = load_data.load_blender_data(args.datadir,args.half_res,args.testskip)
        print('*数据集',args.dataset_type)
        print('*图像尺寸',images.shape,'渲染尺寸',render_poses.shape,'高宽中心',hwf)
        print('*目录',args.datadir)
        i_train,i_val,i_test = i_split
        near = 2.0
        far = 6.0
        if args.white_bkgd: # 数据集blender选项,是否在白色bkgd上渲染
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    else: # 数据集类型未知
        print('数据集类型未知',args.dataset_type)
        return
    H,W,focal = hwf # 数据类型调整
    H,W = int(H),int(W)
    hwf = [H,W,focal] # 高宽中心数值取整数int
    if K is None: # 重新获取hwf高宽中心数值
        K = np.array([[focal,0,0.5*W],[0,focal,0.5*H],[0,0,1]])
    if args.render_test: # 如果要测试,使用测试数据集
        render_poses = np.array(poses[i_test])
    basedir = args.basedir # 存储目录
    expname = args.expname # 项目名称
    os.makedirs(os.path.join(basedir,expname),exist_ok=True) # 创建存储文件夹
    f = os.path.join(basedir,expname,'args.txt') # 打开参数文件args.txt
    with open(f,'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args,arg) # 读取参数信息
            file.write('{} = {}\n'.format(arg,attr)) # 将参数信息写入参数文件args.txt
    if args.config is not None:
        f = os.path.join(basedir,expname,'config.txt') # 打开配置文件config.txt
        with open(f,'w') as file:
            file.write(open(args.config,'r').read()) # 将配置信息写入配置文件config.txt
    print('>>>第二步,训练模型')
    render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer = create_nerf(args) # 创建NeRF模型
    global_step = start # 开始步骤
    bds_dict = {'near':near,'far':far,} # 两个元素
    render_kwargs_train.update(bds_dict) # 训练集九个元素增加两个元素
    render_kwargs_test.update(bds_dict) # 测试集九个元素增加两个元素
    render_poses = torch.Tensor(render_poses).to(device) # 测试数据render_poses加载到设备中
    if args.render_only: # 如果仅渲染并生成视频,使用预渲染模型
        print('*仅渲染模式') # 打印仅渲染模式
        with torch.no_grad(): # 关闭反向传播,禁用梯度计算
            if args.render_test: # 使用测试数据集进行渲染
                images = images[i_test]
            else: # 默认情况渲染更平滑
                images = None
            testsavedir = os.path.join(basedir,expname,'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path',start))
            os.makedirs(testsavedir,exist_ok=True)
            print('*测试数据集渲染尺寸',render_poses.shape)
            rgbs,_ = render_path(render_poses,hwf,K,args.chunk,render_kwargs_test,gt_imgs=images,savedir=testsavedir,render_factor=args.render_factor)
            print('*渲染视频保存地址',testsavedir)
            imageio.mimwrite(os.path.join(testsavedir,'video.mp4'),to8b(rgbs),fps=30,quality=8)
            return
    N_rand = args.N_rand # 梯度步长batch size
    use_batching = not args.no_batching # 是否使用批处理
    if use_batching: # 如果使用批处理,一次读取一批图像
        print('*批处理模式')
        rays = np.stack([get_rays_np(H,W,K,p) for p in poses[:,:3,:4]],0) # 获取相机射线rays[N,ro+rd,H,W,3]
        print('*获取相机射线进行合并')
        rays_rgb = np.concatenate([rays,images[:,None]],1) # 沿axis=1拼接,格式=[N,ro+rd+rgb,H,W,3]
        rays_rgb = np.transpose(rays_rgb,[0,2,3,1,4]) # 改变shape,格式=[N,H,W,ro+rd+rgb,3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train],0) # 只使用训练数据集
        rays_rgb = np.reshape(rays_rgb,[-1,3,3]) # 得到(N-测试样本数目)*H*W个相机射线,格式=[(N-1)*H*W,ro+rd+rgb,3]
        rays_rgb = rays_rgb.astype(np.float32) # 改变相机射线格式
        print('*打乱相机射线顺序')
        np.random.shuffle(rays_rgb) # 打乱相机射线顺序
        print('done')
        i_batch = 0
    if use_batching: # 如果使用批处理,一次读取一批图像
        images = torch.Tensor(images).to(device) # 图像训练数据加载到设备中
    poses = torch.Tensor(poses).to(device) # 位姿训练数据加载到设备中
    if use_batching: # 如果使用批处理,一次读取一批图像
        rays_rgb = torch.Tensor(rays_rgb).to(device) # 射线训练数据加载到设备中
    N_iters = 200000 + 1 # 默认训练200000次(建议修改,加快测试速度)
    print('Begin')
    print('TRAIN views are',i_train)
    print('TEST views are',i_test)
    print('VAL views are',i_val)
    start = start + 1 # 记录训练次数
    print('>>>第三步,体积渲染')
    print('>>>第四步,误差传播')
    print('~~~模型训练函数结束~~~')
# 主函数
if __name__=='__main__':
    print('~~~主函数开始~~~')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
    print('~~~主函数结束~~~')