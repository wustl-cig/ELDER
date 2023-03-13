import torch
import torch.nn as nn
import scipy.io as sio
from os.path import join
from utils import utils_sr
from random import randint,choice
from utils.utils_restoration import matlab_style_gauss2D
import numpy as np
from scipy import ndimage
import cv2
from utils.utils_restoration import array2tensor
from utils import utils_mosaic
from utils import utils_image
class Mri(nn.Module):
    def __init__(self,kernel_path,train_kernel=[0,1,2,3,4],test_kernel=[0,1,2,3,4]) -> None:
        super(Mri,self).__init__()
        kernels=[]
        for cr in range(10,51,10):
            mask = sio.loadmat(join(kernel_path,'mask_%d.mat'%(cr)), squeeze_me=True)['mask_matrix']
            mask = torch.tensor(np.stack([mask, mask], axis=2), dtype=torch.float32)
            kernels.append(mask)
        self.kernels=torch.stack(kernels,dim=0)
        self.val_lst=[{'name':f'val_mri_cs_ratio{10*(k+1)}_psnr','k':k} for k in test_kernel]
        self.kernel_lst=train_kernel
    def make_degradation(self,gt_img,sigma=None,k=None):
        k_idx=choice(self.kernel_lst) if k is None else k
        self.kernel=self.kernels[k_idx].to(gt_img.device)
        degraded_img=torch.view_as_real(torch.fft.fft2(gt_img))*self.kernel
        if len(self.kernel_lst)==1:
            self.sigma=0.0
        else:
            self.sigma=float(5-k_idx)
        self.y=degraded_img.clone()
        return degraded_img
    def make_init(self,degraded_img,tau):
        degraded_img.requires_grad_()
        init_img=torch.fft.ifft2(torch.view_as_complex(degraded_img)).real
        return init_img
    def prox(self,img,tau):
        # tau is step size
        fftz = torch.view_as_real(torch.fft.fft2(img))
        vf=(tau*fftz + self.y )* self.kernel/(1.0+tau) + fftz*(1-self.kernel)
        v = torch.fft.ifft2(torch.view_as_complex(vf)).real
        return v
    def loss(self,current_x,degraded_img):
        deg_x = torch.view_as_complex(torch.view_as_real(torch.fft.fft2(current_x))*self.kernel)
        loss= 0.5*torch.norm(torch.view_as_complex(degraded_img) - deg_x, p=2) ** 2
        return loss
    def val(self,gt_img):
        for val in self.val_lst:
            degraded_img = self.make_degradation(gt_img,k=val['k'])
            yield degraded_img,val['name']