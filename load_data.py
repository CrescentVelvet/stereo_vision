import os
import torch
import numpy
import imageio 
import json
import torch.nn.functional as F
import cv2
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,numpy.cos(phi),-numpy.sin(phi),0],
    [0,numpy.sin(phi),numpy.cos(phi),0],
    [0,0,0,1]]).float()
rot_theta = lambda th : torch.Tensor([
    [numpy.cos(th),0,-numpy.sin(th),0],
    [0,1,0,0],
    [numpy.sin(th),0,numpy.cos(th),0],
    [0,0,0,1]]).float()
def pose_spherical(theta,phi,radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*numpy.pi) @ c2w
    c2w = rot_theta(theta/180.*numpy.pi) @ c2w
    c2w = torch.Tensor(numpy.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w
def load_blender_data(basedir,half_res=False,testskip=1):
    splits = ['train','val','test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir,'transforms_{}.json'.format(s)),'r') as fp:
            metas[s] = json.load(fp)
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir,frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(numpy.array(frame['transform_matrix']))
        imgs = (numpy.array(imgs) / 255.).astype(numpy.float32) # keep all 4 channels (RGBA)
        poses = numpy.array(poses).astype(numpy.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)    
    i_split = [numpy.arange(counts[i],counts[i+1]) for i in range(3)]    
    imgs = numpy.concatenate(all_imgs,0)
    poses = numpy.concatenate(all_poses,0)    
    H,W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / numpy.tan(.5 * camera_angle_x)    
    render_poses = torch.stack([pose_spherical(angle,-30.0,4.0) for angle in numpy.linspace(-180,180,40+1)[:-1]],0)
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.0
        imgs_half_res = numpy.zeros((imgs.shape[0],H,W,4))
        for i,img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img,(W,H),interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res        
    return imgs,poses,render_poses,[H,W,focal],i_split
def load_LINEMOD_data(basedir,half_res=False,testskip=1):
    splits = ['train','val','test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir,'transforms_{}.json'.format(s)),'r') as fp:
            metas[s] = json.load(fp)
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip            
        for idx_test,frame in enumerate(meta['frames'][::skip]):
            fname = frame['file_path']
            if s == 'test':
                print(f"{idx_test}th test frame: {fname}")
            imgs.append(imageio.imread(fname))
            poses.append(numpy.array(frame['transform_matrix']))
        imgs = (numpy.array(imgs) / 255.).astype(numpy.float32) # keep all 4 channels (RGBA)
        poses = numpy.array(poses).astype(numpy.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)    
    i_split = [numpy.arange(counts[i],counts[i+1]) for i in range(3)]    
    imgs = numpy.concatenate(all_imgs,0)
    poses = numpy.concatenate(all_poses,0)    
    H,W = imgs[0].shape[:2]
    focal = float(meta['frames'][0]['intrinsic_matrix'][0][0])
    K = meta['frames'][0]['intrinsic_matrix']
    print(f"Focal: {focal}")    
    render_poses = torch.stack([pose_spherical(angle,-30.0,4.0) for angle in numpy.linspace(-180,180,40+1)[:-1]],0)
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.0
        imgs_half_res = numpy.zeros((imgs.shape[0],H,W,3))
        for i,img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img,(W,H),interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    near = numpy.floor(min(metas['train']['near'],metas['test']['near']))
    far = numpy.ceil(max(metas['train']['far'],metas['test']['far']))
    return imgs,poses,render_poses,[H,W,focal],K,i_split,near,far