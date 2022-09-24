# NeRF神经辐射场
import os
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 获取射线函数
def get_rays(H,W,K,c2w):
    i,j = torch.meshgrid(torch.linspace(0,W-1,W),torch.linspace(0,H-1,H)) # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0],-(j-K[1][2])/K[1][1],-torch.ones_like(i)],-1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[...,np.newaxis,:] * c2w[:3,:3],-1)  # dot product,equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o,rays_d
# 批处理函数
def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret
# 射线批处理函数
def batchify_rays(rays_flat, # [4096,8]类型
                  chunk=1024*32, # 同时处理的最大射线数,用于防止内存溢出,int类型
                  **kwargs): # 渲染一批射线
    all_ret = {}
    for i in range(0,rays_flat.shape[0],chunk):
        ret = render_rays(rays_flat[i:i+chunk],**kwargs) # 渲染一批射线
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k : torch.cat(all_ret[k],0) for k in all_ret}
    return all_ret # 返回一批射线渲染结果
# 分层抽样函数
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))
    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)
    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    return samples
# 射线渲染函数
def render_rays(ray_batch, # 一批射线的原点方向数组,[batch_size,...]类型
                network_fn, # 用于预测空间中每个点颜色和密度的粗网络模型
                network_query_fn, # 用于将查询传递给network的函数
                N_samples, # 每条射线的粗采样点数
                retraw=False, # 是否包含模型原始预测
                lindisp=False, # 是否在逆深度而不是深度中线性采样
                perturb=0.0, # 非0表示在时间上分层随机点对射线采样
                N_importance=0, # 每条射线附加的细采样点数
                network_fine=None, # 用于预测空间中每个点颜色和密度的细网络模型
                white_bkgd=False, # 是否假设背景为白色
                raw_noise_std=0.0, # 是否加入噪音方差
                verbose=False, # 是否打印更多调试信息
                pytest=False): # 是否使用numpy的固定随机数
    N_rays = ray_batch.shape[0] # 提取一批射线数量
    rays_o,rays_d = ray_batch[:,0:3],ray_batch[:,3:6] # 提取射线的原点和方向,[N_rays,3]类型
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None # 
    bounds = torch.reshape(ray_batch[...,6:8],[-1,1,2]) # 
    near,far = bounds[...,0],bounds[...,1] # 提取射线的最近和最远距离
    t_vals = torch.linspace(0.0,1.0,steps=N_samples) # 每条射线上粗采样N_samples个点
    if not lindisp: # 如果在逆深度而不是深度中线性采样
        z_vals = near*(1.-t_vals) + far*(t_vals)
    else: # 否则在深度中线性采样
        z_vals = 1.0/(1.0/near*(1.0-t_vals) + 1.0/far*(t_vals))
    z_vals = z_vals.expand([N_rays,N_samples]) # 元素复制多份,数组扩展为[N_rays射线数,N_samples]类型
    if perturb > 0.0: # 如果在时间上分层随机点对射线采样
        mids = 0.5*(z_vals[...,1:] + z_vals[...,:-1]) # 提取采样点之间的间隔
        upper = torch.cat([mids,z_vals[...,-1:]],-1) # 提取前半采样点
        lower = torch.cat([z_vals[...,:1],mids],-1) # 提取后半采样点
        t_rand = torch.rand(z_vals.shape) # 在间隔内随机分层样本
        if pytest: # 如果使用numpy的固定随机数替代随机分层样本
            np.random.seed(0) # 生成随机数种子
            t_rand = np.random.rand(*list(z_vals.shape)) # 生成随机数替代随机分层样本
            t_rand = torch.Tensor(t_rand) # 转换为张量
        z_vals = lower + (upper - lower)*t_rand # 采样点映射到小区间内
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # 提取采样点,[N_rays射线数=4096,N_samples=64,3]类型
    raw = network_query_fn(pts,viewdirs,network_fn) # 使用pts和viewdirs进行前向计算,[N_rays射线数=4096,N_samples=64,RGB+密度=4]类型
    rgb_map,disp_map,acc_map,weights,depth_map = raw2outputs(raw,z_vals,rays_d,raw_noise_std,white_bkgd,pytest=pytest) # 将射线颜色绘制成图像上的点
    if N_importance > 0: # 如果每条射线附加的细采样点数大于0,重复一遍细采样步骤
        rgb_map_0,disp_map_0,acc_map_0 = rgb_map,disp_map,acc_map # 保存之前计算的射线预测颜色,视差图像,累加密度
        z_vals_mid = 0.5 * (z_vals[...,1:] + z_vals[...,:-1]) # 重新采样射线上的点
        z_samples = sample_pdf(z_vals_mid,weights[...,1:-1],N_importance,det=(perturb==0.),pytest=pytest)
        z_samples = z_samples.detach()
        z_vals,_ = torch.sort(torch.cat([z_vals,z_samples],-1),-1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays射线数,N_samples+N_importance,3]类型
        run_fn = network_fn if network_fine is None else network_fine # 使用细采样或粗采样模型
        raw = network_query_fn(pts,viewdirs,run_fn)
        rgb_map,disp_map,acc_map,weights,depth_map = raw2outputs(raw,z_vals,rays_d,raw_noise_std,white_bkgd,pytest=pytest) # 将射线颜色绘制成图像上的点
    ret = {'rgb_map' : rgb_map,'disp_map' : disp_map,'acc_map' : acc_map} # 采样结果封装到ret,不论有无细采样
    if retraw: # 如果包含模型原始预测
        ret['raw'] = raw # 封装raw
    if N_importance > 0: # 如果每条射线附加的细采样点数大于0
        ret['rgb0'] = rgb_map_0 # 射线的预测颜色
        ret['disp0'] = disp_map_0 # 射线的视差图像,深度的倒数
        ret['acc0'] = acc_map_0 # 射线的累积不透明度,累加密度
        ret['z_std'] = torch.std(z_samples,dim=-1,unbiased=False) # [N_rays]类型
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret # 返回封装的射线渲染结果
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
# 预测转换意义函数
def raw2outputs(raw, # 模型预测结果,[num_rays,num_samples along ray,4]类型
                z_vals, # 积分时间,[num_rays,num_samples along ray]类型
                rays_d, # 每条射线的方向,[num_rays,3]类型
                raw_noise_std=0, # 是否加入噪音方差
                white_bkgd=False, # 是否假设背景为白色
                pytest=False): # 是否使用numpy的固定随机数
    raw2alpha = lambda raw,dists,act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists,torch.Tensor([1e10]).expand(dists[...,:1].shape)],-1) # [N_rays,N_samples]类型
    dists = dists * torch.norm(rays_d[...,None,:],dim=-1)
    rgb = torch.sigmoid(raw[...,:3])  # 获取模型预测每个点的颜色,[N_rays,N_samples,3]类型
    noise = 0.0 # 噪声初始化
    if raw_noise_std > 0.0: # 如果加入噪音方差
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
        if pytest: # 如果使用numpy的固定随机数替代随机样本
            np.random.seed(0) # 生成随机数种子
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std # 生成随机噪音
            noise = torch.Tensor(noise) # 噪音转换为张量
    alpha = raw2alpha(raw[...,3] + noise,dists)  # 给密度添加噪音,[N_rays,N_samples]类型
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0],1)),1.-alpha + 1e-10],-1),-1)[:,:-1] # 射线分配给每个采用颜色的权重,[N_rays,N_samples]类型
    rgb_map = torch.sum(weights[...,None] * rgb,-2)  # 射线的预测颜色,[N_rays,3]类型
    depth_map = torch.sum(weights * z_vals,-1) # 射线的预测深度,到目标的估计距离,[N_rays]类型
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),depth_map / torch.sum(weights,-1)) # 射线的视差图像,深度的倒数,[N_rays]类型
    acc_map = torch.sum(weights,-1) # 射线的累积不透明度,累加密度,[N_rays]类型
    if white_bkgd: # 如果假设背景为白色
        rgb_map = rgb_map + (1.0-acc_map[...,None])
    return rgb_map,disp_map,acc_map,weights,depth_map # 返回模型预测转换为具有语义意义的值
# 网络运行函数
def run_network(inputs,
                viewdirs,
                fn,
                embed_fn,
                embeddirs_fn,
                netchunk=1024*64):
    inputs_flat = torch.reshape(inputs,[-1,inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs,[-1,input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded,embedded_dirs],-1)
    outputs_flat = batchify(fn,netchunk)(embedded)
    outputs = torch.reshape(outputs_flat,list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs
# 模型建立函数
def create_nerf(args):
    print('-开始建立模型')
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
        print('-读取已保存权重',ckpts)
    if len(ckpts) > 0 and not args.no_reload: # 如果要从保存的模型中加载权重
        ckpt_path = ckpts[-1] # 把checkpoints数组[-1]作为路径
        print('-从保存的模型中加载权重',ckpt_path)
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
        render_kwargs_train['ndc'] = False # 不使用ndc
        render_kwargs_train['lindisp'] = args.lindisp
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.0
    print('-完成建立模型')
    return render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer
# 渲染函数
def render(H, # 图像高度,单位是像素,int类型
           W, # 图像宽度,单位是像素,int类型
           K, # 相机焦距,float类型
           chunk=1024*32, # 同时处理的最大射线数,用于防止内存溢出,int类型
           rays=None, # 相机射线数组,射线起点和方向一一对应,[2,batch_size,3]类型
           c2w=None, # 相机世界转换矩阵,[3,4]类型
           ndc=True, # 是否表示原始射线,即NDC坐标中的方向,bool类型
           near=0.0, # 射线的最近距离,float或[batch_size]类型
           far=1.0, # 射线的最远距离,float或[batch_size]类型
           use_viewdirs=False, # 是否输入射线方向,bool类型
           c2w_staticcam=None, # 相机世界转换矩阵,[3,4]类型
           **kwargs): # 体积渲染函数返回射线对应的颜色,视差,不透明度
    if c2w is not None: # 如果是渲染完整图像的特殊情况
        rays_o,rays_d = get_rays(H,W,K,c2w) # 像素点坐标方向提取
    else: # 否则使用提供的射线batch批处理
        rays_o,rays_d = rays # 像素点坐标方向直接赋值
    if use_viewdirs: # 如果输入射线方向
        viewdirs = rays_d
        if c2w_staticcam is not None: # 可视化viewdir效果的特例
            rays_o,rays_d = get_rays(H,W,K,c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs,dim=-1,keepdim=True)
        viewdirs = torch.reshape(viewdirs,[-1,3]).float()
    sh = rays_d.shape # sh[4096,3]类型
    if ndc: # 如果是前向场景
        rays_o,rays_d = ndc_rays(H,W,K[0][0],1.0,rays_o,rays_d) # 像素点坐标方向使用ndc转换
    rays_o = torch.reshape(rays_o,[-1,3]).float() # 创建射线batch批处理
    rays_d = torch.reshape(rays_d,[-1,3]).float()
    near,far = near * torch.ones_like(rays_d[...,:1]),far * torch.ones_like(rays_d[...,:1]) # near[4096,1]类型,far[4096,1]类型,全0或全1
    rays = torch.cat([rays_o,rays_d,near,far],-1) # rays[4096,3+3+1+1=8]类型
    if use_viewdirs: # 如果输入射线方向
        rays = torch.cat([rays,viewdirs],-1)
    all_ret = batchify_rays(rays,chunk,**kwargs) # 渲染一批射线,chunk默认值1024*32=32768
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k],k_sh)
    k_extract = ['rgb_map', # 射线的预测颜色,[batch_size,3]类型
                 'disp_map', # 射线的视差图像,深度的倒数,[batch_size]类型
                 'acc_map'] # 射线的累积不透明度,密累加度,[batch_size]类型
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]
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
            print('*测试数据render_poses渲染尺寸',render_poses.shape)
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
        np.random.shuffle(rays_rgb) # 打乱相机射线顺序
        print('*打乱相机射线顺序')
        i_batch = 0
        images = torch.Tensor(images).to(device) # 图像训练数据加载到设备中
        rays_rgb = torch.Tensor(rays_rgb).to(device) # 射线训练数据加载到设备中
    poses = torch.Tensor(poses).to(device) # 位姿训练数据加载到设备中
    N_iters = 512 + 1 # 默认训练200000次(建议修改,加快测试速度)
    print('*训练集',i_train)
    print('*测试集',i_test)
    print('*验证集',i_val)
    start = start + 1 # 记录训练次数
    print('*开始第',start,'次训练')
    for i in tqdm.trange(start,N_iters): # 循环训练N_iters次
        time0 = time.time() # 每次训练开始时间
        if use_batching: # 如果使用批处理,一次读取一批图像
            batch = rays_rgb[i_batch:i_batch+N_rand] # 从相机射线中取一个batch,格式=[B,2+1,3*?]
            batch = torch.transpose(batch,0,1) # 转换0维和1维数据
            batch_rays,target_s = batch[:2],batch[2] # batch_rays格式=[ro+rd,4096,3],target_s格式=[4096,3]对应的是rgb
            i_batch += N_rand # 一次增加一个梯度步长batch size
            if i_batch >= rays_rgb.shape[0]: # 如果所有样本都遍历过了则打乱数据顺序
                print("*经过一个epoch后打乱数据顺序")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else: # 如果不使用批处理,一次仅读取一张图像
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i,:3,:4]
            if N_rand is not None: # 如果梯度步长batch size存在
                rays_o,rays_d = get_rays(H,W,K,torch.Tensor(pose)) # 获取相机射线(H,W,3),(H,W,3)
                if i < args.precrop_iters: # 当i小于中心crop的训练步数时
                    dH = int(H//2*args.precrop_frac)
                    dW = int(W//2*args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2-dH,H//2+dH-1,2*dH),
                            torch.linspace(W//2-dW,W//2+dW-1,2*dW)
                        ),-1)
                    if i == start:
                        print(f'*在训练{args.precrop_iters}次之前进行大小为{2*dH}x{2*dW}的中心裁剪')                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0,H-1,H),torch.linspace(0,W-1,W)),-1) # (H,W,2)
                coords = torch.reshape(coords,[-1,2]) # (H * W,2)
                select_inds = np.random.choice(coords.shape[0],size=[N_rand],replace=False) # (N_rand,)
                select_coords = coords[select_inds].long() # (N_rand,2)
                rays_o = rays_o[select_coords[:,0],select_coords[:,1]] # (N_rand,3)
                rays_d = rays_d[select_coords[:,0],select_coords[:,1]] # (N_rand,3)
                batch_rays = torch.stack([rays_o,rays_d],0)
                target_s = target[select_coords[:,0],select_coords[:,1]] # (N_rand,3)
        # print('>>>第三步,体积渲染')
        rgb,disp,acc,extras = render(H,W,K, # 返回渲染一批的颜色,视差图,不透明度,其他信息
                                     chunk=args.chunk, # 同时处理的最大射线数,用于防止内存溢出,int类型
                                     rays=batch_rays, # 一批相机射线
                                     verbose=i < 10, # 是否打印更多调试信息
                                     retraw=True,
                                     **render_kwargs_train)
        # print('>>>第四步,误差传播')
        img2mse = lambda x,y : torch.mean((x-y)**2) # MSE均方误差损失函数
        mse2psnr = lambda x : -10.0*torch.log(x)/torch.log(torch.Tensor([10.0])) # PSNR信噪比函数
        to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8) # 
        optimizer.zero_grad() # 清空优化器历史梯度
        img_loss = img2mse(rgb,target_s) # 计算MSE均方误差损失
        trans = extras['raw'][...,-1]
        loss = img_loss # 损失记为MSE均方误差损失
        psnr = mse2psnr(img_loss) # 计算PSNR信噪比
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'],target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        loss.backward() # 误差反向传播
        optimizer.step() # 更新优化器参数
        decay_rate = 0.1 # 设置学习率0.1
        decay_steps = args.lrate_decay * 1000 # 设置指数学习率衰减
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps)) # 设置每次训练使用衰减学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate # 修改优化器参数组中的学习率
        dt = time.time()-time0 # 计算训练用时
        if i % args.i_weights == 0: # 按频率保存checkpoint
            path = os.path.join(basedir,expname,'{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },path)
            print('Saved checkpoints at',path) # 打印保存checkpoint
        if i % args.i_video == 0 and i > 0: # 按频率保存mp4渲染视频
            with torch.no_grad(): # 关闭反向传播,禁用梯度计算
                rgbs,disps = render_path(render_poses,hwf,K,args.chunk,render_kwargs_test)
            print('Done,saving',rgbs.shape,disps.shape)
            moviebase = os.path.join(basedir,expname,'{}_spiral_{:06d}_'.format(expname,i))
            imageio.mimwrite(moviebase + 'rgb.mp4',to8b(rgbs),fps=30,quality=8) # 保存颜色视频
            imageio.mimwrite(moviebase + 'disp.mp4',to8b(disps / np.max(disps)),fps=30,quality=8) # 保存深度视频
        if i % args.i_testset == 0 and i > 0: # 按频率保存测试数据集
            testsavedir = os.path.join(basedir,expname,'testset_{:06d}'.format(i))
            os.makedirs(testsavedir,exist_ok=True)
            print('test poses shape',poses[i_test].shape)
            with torch.no_grad(): # 关闭反向传播,禁用梯度计算
                render_path(torch.Tensor(poses[i_test]).to(device),hwf,K,args.chunk,render_kwargs_test,gt_imgs=images[i_test],savedir=testsavedir)
            print('Saved test set')
        if i % args.i_print==0: # 按频率打印输出和日志
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname,i,psnr.numpy(),loss.numpy(),global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss',loss)
                tf.contrib.summary.scalar('psnr',psnr)
                tf.contrib.summary.histogram('tran',trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0',psnr0)
            if i % args.i_img == 0: # 按频率在tensorboard绘制图像
                img_i = np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i,:3,:4]
                with torch.no_grad():
                    rgb,disp,acc,extras = render(H,W,focal,chunk=args.chunk,c2w=pose,**render_kwargs_test)
                psnr = mse2psnr(img2mse(rgb,target))
                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                    tf.contrib.summary.image('rgb',to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp',disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc',acc[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.scalar('psnr_holdout',psnr)
                    tf.contrib.summary.image('rgb_holdout',target[tf.newaxis])
                if args.N_importance > 0:
                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0',to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0',extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std',extras['z_std'][tf.newaxis,...,tf.newaxis])
        """
        global_step += 1 # 当前步骤加一
    print('~~~模型训练函数结束~~~')
# 主函数
if __name__=='__main__':
    print('~~~主函数开始~~~')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
    print('~~~主函数结束~~~')