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
from utils import utils_mosaic
from utils import utils_image
class Inpainting(nn.Module):
    def __init__(self,mask_probs) -> None:
        super(Inpainting,self).__init__()
        self.mask_probs=mask_probs
        self.val_lst=[{'name':f'val_inpainting_prob={prob}','prob':prob} for prob in self.mask_probs]

    def make_degradation(self,gt_img,sigma=None,prob=None,seed=None):
        p=prob if prob is not None else choice(self.mask_probs)
        self.sigma=10*p
        if seed is not None:
            torch.random.manual_seed(seed)
        self.kernel=torch.bernoulli(torch.tensor(1-p,dtype=torch.float32,device=gt_img.device).expand(gt_img.shape[0],1,gt_img.shape[2],gt_img.shape[3])).expand(gt_img.shape[0],3,gt_img.shape[2],gt_img.shape[3])
        degraded_img=gt_img*self.kernel + (0.5)*(1-self.kernel)
        self.masked_img=self.kernel*degraded_img
        return degraded_img
            
        
    def make_init(self,degraded_img,tau):
        degraded_img.requires_grad_()
        return self.prox(degraded_img,tau)
    def prox(self,img,tau):
        # tau is step size
        proxima = self.masked_img + (1-self.kernel)*img
        return proxima
    def loss(self,current_x,degraded_img):
        deg_x = self.kernel*current_x #+ (0.5)*(1-self.kernel)
        loss= 0.5*torch.norm(degraded_img*self.kernel - deg_x, p=2) ** 2
        return loss
    def val(self,gt_img):
        for val in self.val_lst:
            degraded_img = self.make_degradation(gt_img,prob=val['prob'])
            yield degraded_img,val['name']