import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import logging
from models.dpir_unet import DPIRNNclass
from utils.utils_image import tensor2single
from models.tasks import inpainting,mri,super_resolution
import cv2
from scipy.optimize import fminbound,fmin,brute
from omegaconf import DictConfig
from torchvision.transforms.functional import resize,InterpolationMode
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
logger = logging.getLogger(__name__)

def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=0.0):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))
    rhos = list(map(lambda x: 0.23*(sigma**2)/(x**2), sigmas/255.0))
    return rhos, sigmas

class Restore(nn.Module):
    def __init__(self, regularization,
                task,
                init_tau,
                init_lamb,
                init_sigma,
                backtracking=False,
                max_iter=100,
                tol=1e-5,
                enable_diff=False,
                relative_diff_F_min=1e-6,
                inpainting_init=False,
                n_init=10,
                backtracking_max_try=None,
                gamma=0.1,
                eta_tau=0.5,
                accelerate=False,
                exp_sigma_schedule=False):
        super(Restore, self).__init__()
        self.regularization=regularization
        self.task=task
        self.init_tau=init_tau
        self.init_lamb=init_lamb
        self.init_sigma=init_sigma
        self.backtracking=backtracking
        self.max_iter=max_iter
        self.tol=tol
        self.enable_diff=enable_diff
        self.relative_diff_F_min=relative_diff_F_min
        self.inpainting_init=inpainting_init
        self.n_init=n_init
        self.backtracking_max_try=backtracking_max_try if backtracking_max_try is not None else np.inf
        self.gamma=gamma
        self.eta_tau=eta_tau
        self.accelerate=accelerate
        self.exp_sigma_schedule = exp_sigma_schedule
    def modcrop(self,img_in, scale):
    # img_in: Numpy, HWC or HW
        img = img_in
        _,_,H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:,:,:H - H_r, :W - W_r]
        return img
    def forward(self, gt_img,aux,tau=None,lamb=None,sigma_denoiser=None,extract_results=False,log='debug',max_iter=None):
        if log=='info':
            logger.setLevel(logging.INFO)
        elif log=='debug':
            logger.setLevel(logging.DEBUG)
        
        results={'psnr':[],'loss':[],'z':[],'x':[], 'Dx':[],'tau':[],'pure_loss':[],'h':[],'g':[]}
        tau=self.init_tau if tau is None else tau
        lamb=self.init_lamb if lamb is None else lamb
        sigma_denoiser = self.init_sigma if sigma_denoiser is None else sigma_denoiser
        if self.exp_sigma_schedule:
            _ , sigmas = get_rho_sigma(sigma=2.55/255., iter_num=self.max_iter,modelSigma1= sigma_denoiser, modelSigma2= 7.65)
        tau=torch.tensor(tau,dtype=gt_img.dtype,device=gt_img.device)
        if isinstance(self.task,super_resolution.SuperResolution):
            gt_img=self.modcrop(gt_img,aux['sf'])
        degraded_img=self.task.make_degradation(gt_img,**aux)
        if not isinstance(self.regularization, DPIRNNclass):
            if isinstance(self.task,super_resolution.SuperResolution):
                f_init = self.regularization.calculate_grad(resize(degraded_img,(self.task.scale_factor*degraded_img.shape[-2],self.task.scale_factor*degraded_img.shape[-1]),interpolation=InterpolationMode.BICUBIC),sigma_denoiser/255.)[1]
                s_init = self.regularization.loss(f_init,resize(degraded_img,(self.task.scale_factor*degraded_img.shape[-2],self.task.scale_factor*degraded_img.shape[-1]),interpolation=InterpolationMode.BICUBIC))
                g_init = self.task.loss(resize(degraded_img,(self.task.scale_factor*degraded_img.shape[-2],self.task.scale_factor*degraded_img.shape[-1]),interpolation=InterpolationMode.BICUBIC),degraded_img)
                F_init = g_init + lamb * s_init.item()
                F_pure_init = g_init + s_init.item()
            elif isinstance(self.task,mri.Mri):
                f_init = self.regularization.calculate_grad(torch.fft.ifft2(torch.view_as_complex(degraded_img)).real.clamp(0,1),sigma_denoiser/255.)[1]
                s_init = self.regularization.loss(f_init,torch.fft.ifft2(torch.view_as_complex(degraded_img)).real.clamp(0,1))
                g_init = self.task.loss(torch.fft.ifft2(torch.view_as_complex(degraded_img)).real.clamp(0,1),degraded_img)
                F_init = g_init + lamb * s_init.item()
                F_pure_init = g_init + s_init.item()
            elif isinstance(self.task,inpainting.Inpainting):
                f_init = self.regularization.calculate_grad(degraded_img,sigma_denoiser/255.)[1]
                s_init = self.regularization.loss(f_init,degraded_img)
                g_init = self.task.loss(degraded_img.clamp(0,1),degraded_img)
                F_init = g_init + lamb * s_init.item()
                F_pure_init = g_init + s_init.item()
            results['loss'].append(F_init.item())
            results['pure_loss'].append(F_pure_init.item())
            results['h'].append(s_init.item())
            results['g'].append(g_init.item())
        x0=self.task.make_init(degraded_img,tau)
        if not isinstance(self.task,mri.Mri):
            results['degraded_img']=tensor2single(degraded_img)
        else:
            results['degraded_img']=tensor2single(x0)
        iter_times=0
        peak_psnr=sk_psnr(gt_img.clamp(0,1).detach().cpu().numpy(),x0.clamp(0,1).detach().cpu().numpy()).item()
        logger.debug(f'init_psnr:{peak_psnr}')
        if extract_results:
            results['psnr'].append(peak_psnr)
        peak_iteration=iter_times
        x=x0
        diff_F = 1
        F_old = 1
        relative_diff_F_min = self.relative_diff_F_min
        if max_iter is None:
            max_iter = self.max_iter
        # results['tau'].append(tau.item())
        s_x = x  # gradient update
        t = torch.tensor(1., dtype=torch.float32).to(x.device)
        while iter_times < max_iter:
            if self.enable_diff==True and abs(diff_F)/abs(F_old) < relative_diff_F_min and not isinstance(self.regularization,DPIRNNclass):
                logger.debug(f'iteration meet the convergence requirement. iter_times:{iter_times},diff_F:{diff_F},F_old:{F_old}')
                break
            if self.inpainting_init :
                if iter_times < self.n_init:
                    sigma_denoiser = 50
                    relative_diff_F_min = 0
                else :
                    sigma_denoiser = self.init_sigma if sigma_denoiser is None else sigma_denoiser
                    relative_diff_F_min = self.relative_diff_F_min
            else :
                if self.exp_sigma_schedule:
                    sigma_denoiser = sigmas[iter_times]
                else:
                    sigma_denoiser = self.init_sigma if sigma_denoiser is None else sigma_denoiser
            x_old=x
            # if log == 'debug':
            #     sigma_denoiser = fmin(lambda x: -sk_psnr(gt_img,s_x - self.regularization.calculate_grad(s_x.clamp(min=0.0),x[0]/255.)[0].detach()).item(),[sigma_denoiser.item(),],disp=False,ftol=1e-2,maxfun=5)[0]
            #     logger.debug(f'iter_times:{iter_times},sigma_denoiser:{sigma_denoiser}')
            Ds, f=self.regularization.calculate_grad(s_x.clamp(min=0.0),sigma_denoiser/255.)
            Ds = Ds.detach()
            f = f.detach()
            Dx = s_x -  Ds
            denoised_psnr=sk_psnr(gt_img.clamp(0,1).detach().cpu().numpy(),Dx.clamp(0,1).detach().cpu().numpy()).item()
            logger.debug(f'iter_times:{iter_times},denoised_psnr:{denoised_psnr}')
            if not isinstance(self.regularization,DPIRNNclass):
                s_old=self.regularization.loss(f,s_x)
                g_old=self.task.loss(s_x.clamp(0,1),degraded_img)
                F_old = g_old + lamb * s_old.item()
            backtracking_check = False
            bt_times = 0
            while not backtracking_check:
                z = (1 - lamb * tau) * x_old + lamb * tau * Dx
                x=self.task.prox(z,tau)
                temp_psnr=sk_psnr(gt_img.clamp(0,1).detach().cpu().numpy(),x.clamp(0,1).detach().cpu().numpy()).item()
                logger.debug(f'iter_times:{iter_times}, bt_times: {bt_times},temp_psnr:{temp_psnr}')
                if not isinstance(self.regularization,DPIRNNclass):
                    f=self.regularization.calculate_grad(x,sigma_denoiser/255.)[1]
                    s=self.regularization.loss(f,x)
                    g=self.task.loss(x.clamp(0,1),degraded_img)
                    F_new =  g + lamb * s.item()
                    F_pure_new = g + s.item()
                    diff_x = (torch.norm(x - x_old, p=2) ** 2).item()
                    diff_F = (F_old - F_new).item()
                    if self.backtracking and diff_F < (self.gamma / tau) * diff_x and abs(diff_F)/abs(F_old) > relative_diff_F_min and bt_times<self.backtracking_max_try:
                        backtracking_check = False
                        tau = self.eta_tau * tau
                        x = x_old
                        bt_times+=1
                    else:
                        backtracking_check = True
                else:
                    backtracking_check = True
            x = x.clamp(min=0.0)
            if self.accelerate:
                tnext = 0.5*(1+torch.sqrt(1+4*t*t))
            else:
                tnext = 1
            s_x = x + ((t-1)/tnext)*(x-x_old)
            
            # update
            t = tnext

            if not isinstance(self.task,super_resolution.SuperResolution):
                curr_psnr=sk_psnr(gt_img.clamp(0,1).detach().cpu().numpy(),x.clamp(0,1).detach().cpu().numpy()).item()
            else:
                curr_psnr=sk_psnr(gt_img.clamp(0,1).detach().cpu().numpy(),self.regularization(x,sigma_denoiser/255.,create_graph=False,strict=False).clamp(0,1).detach().cpu().numpy()).item()
            logger.debug(f'iter_times:{iter_times},curr_psnr:{curr_psnr}')
            if curr_psnr>peak_psnr:
                peak_psnr=curr_psnr
                peak_iteration=iter_times
                if not isinstance(self.task,super_resolution.SuperResolution):
                    results['peak_image']=tensor2single(x)
                else:
                    results['peak_image']=tensor2single(self.regularization(x,sigma_denoiser/255.,create_graph=False,strict=False))
            results['tau'].append(tau.item())
            if extract_results:
                results['psnr'].append(curr_psnr)
                if not isinstance(self.regularization,DPIRNNclass):
                    results['loss'].append(F_new.item())
                    results['pure_loss'].append(F_pure_new.item())
                    results['h'].append(s.item())
                    results['g'].append(g.item())
                results['z'].append(tensor2single(z))
                results['x'].append(tensor2single(x))
                results['Dx'].append(tensor2single(Dx))
            iter_times+=1
        Ds, f=self.regularization.calculate_grad(x,sigma_denoiser/255.)
        Ds = Ds.detach()
        f = f.detach()
        Dx = x - Ds
        if not isinstance(self.regularization,DPIRNNclass):
            s=self.regularization.loss(f,x)
        z = (1 - lamb * tau) * x + lamb * tau * Dx
        if not isinstance(self.task,super_resolution.SuperResolution):
            output_img=x
        else:
            output_img=Dx
        final_psnr=sk_psnr(gt_img.clamp(0,1).detach().cpu().numpy(),output_img.clamp(0,1).detach().cpu().numpy()).item()
        if final_psnr>peak_psnr:
            peak_psnr=final_psnr
            peak_iteration=max_iter
        logger.debug(f'final_psnr:{final_psnr}')
        results['recon_image']=tensor2single(output_img)
        results['final_psnr']=final_psnr
        results['peak_psnr']=peak_psnr
        results['peak_iteration']=peak_iteration
        return results
    @staticmethod
    def batch_psnr(pred,gt):
        pred=pred.clamp(0,1)
        gt=gt.clamp(0,1)
        mse=torch.mean((pred-gt)**2,dim=[1,2,3])
        return (20*torch.log10(1. /torch.sqrt(mse))).mean(0)
