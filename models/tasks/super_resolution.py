from itertools import product
import torch
import torch.nn as nn
from hdf5storage import loadmat
from utils import utils_sr
from random import randint,choice
from utils.utils_restoration import matlab_style_gauss2D
import numpy as np
from scipy import ndimage
import cv2
from utils.utils_restoration import array2tensor
class SuperResolution(nn.Module):
    def __init__(self,
                kernel_path,
                scale_factors,
                test_scale_factors,
                test_noise_levels,
                test_kernels) -> None:
        super(SuperResolution,self).__init__()
        self.kernels=loadmat(kernel_path)['kernels']
        self.scale_factors=scale_factors
        self.kernel_type='motion' if 'Levin09.mat' in kernel_path else 'gaussian'
        self.val_lst=[{'name':f'val_super-resolution_k={k_idx}_sf={sf}_sigma={sigma}','k_idx':k_idx,'sf':sf,'sigma':sigma} for k_idx,sf,sigma in product(test_kernels,test_scale_factors,test_noise_levels)]
    def make_degradation(self,gt_img,sigma,k_idx=None,sf=None,seed=None):
        self.sigma=sigma
        if self.kernel_type=='motion':
            kernel_idx=randint(0,9) if k_idx is None else k_idx
            if kernel_idx == 8: # Uniform blur
                self.kernel = (1/81)*np.ones((9,9))
            elif kernel_idx == 9:  # Gaussian blur
                self.kernel = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
            else: # Motion blur
                self.kernel = self.kernels[0, kernel_idx]
        else:
            kernel_idx=randint(0,9) if k_idx is None else k_idx
            self.kernel= self.kernels[0, kernel_idx]
        self.scale_factor=choice(self.scale_factors) if sf is None else sf
        if seed is not None:
            np.random.seed(seed)
        degradLst=[ndimage.filters.convolve(self.modcrop(gt_img[i,...].permute(1,2,0).cpu().numpy(),self.scale_factor), np.expand_dims(self.kernel, axis=2), mode='wrap')[0::self.scale_factor,0::self.scale_factor,...]+np.random.normal(0, sigma/255., (gt_img.shape[2]//self.scale_factor,gt_img.shape[3]//self.scale_factor,gt_img.shape[1])) for i in range(gt_img.shape[0])]
        degraded_img=torch.tensor(np.stack(degradLst,axis=0),dtype=torch.float32,device=gt_img.device).permute(0,3,1,2)
        self.kernal_tensor = array2tensor(np.expand_dims(self.kernel, 2)).float().to(degraded_img.device)
        return degraded_img
    def make_init(self,degraded_img,tau):
        rescaledLst=[utils_sr.shift_pixel(cv2.resize(degraded_img[i,...].detach().permute(1,2,0).cpu().numpy(), (degraded_img.shape[3] * self.scale_factor, degraded_img.shape[2] * self.scale_factor),interpolation=cv2.INTER_CUBIC),self.scale_factor) for i in range(degraded_img.shape[0])]
        rescaled=torch.tensor(np.stack(rescaledLst,axis=0),dtype=torch.float32,device=degraded_img.device).permute(0,3,1,2)
        rescaled.requires_grad_()
        
        self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate(degraded_img, self.kernal_tensor, self.scale_factor)
        return self.prox(rescaled,tau)
    def prox(self,img,tau):
        # tau is step size
        proxima = utils_sr.data_solution(img, self.FB, self.FBC, self.F2B, self.FBFy, alpha=1/tau, sf=self.scale_factor)
        return proxima
    def modcrop(self,img_in, scale):
    # img_in: Numpy, HWC or HW
        img = img_in
        if img.ndim == 2:
            H, W = img.shape
            H_r, W_r = H % scale, W % scale
            img = img[:H - H_r, :W - W_r]
        elif img.ndim == 3:
            H, W, C = img.shape
            H_r, W_r = H % scale, W % scale
            img = img[:int(H-H_r), :int(W-W_r), :]
        else:
            raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
        return img
    def loss(self,current_x,degraded_img):
        deg_x = utils_sr.imfilter(current_x, self.kernal_tensor[0].flip(1).flip(2).expand(3, -1, -1, -1))
        deg_x = deg_x[...,0::self.scale_factor, 0::self.scale_factor]
        loss = 0.5 * torch.norm(degraded_img - deg_x, p=2) ** 2
        return loss
    def val(self,gt_img):
        for val in self.val_lst:
            degraded_img=self.make_degradation(gt_img,sigma=val['sigma'],k_idx=val['k'],sf=val['sf'])
            yield degraded_img,val['name']