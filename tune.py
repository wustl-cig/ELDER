import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate,get_original_cwd
import pytorch_lightning as pl
import torch
from os.path import join,isabs
from os import listdir
import logging
from natsort import natsorted
from models.dpir_unet import DPIRNNclass,GSPNPNNclass,REDPotentialNNclass,PotentialNNclass
from models.tasks import mri,inpainting,super_resolution
from utils.utils_restore import Restore
from scipy.optimize import fminbound,fmin,brute
from pathlib import Path
import pandas as pd
import numpy as np
from torchvision.transforms.functional import to_tensor
from PIL.Image import open as open_img
from utils.utils_restoration import imsave,single2uint
from utils.utils_image import tensor2single
from utils.utils_mosaic import dm_matlab
from torch.nn.functional import pad
from os.path import isdir

logger = logging.getLogger(__name__)

def prefix_path(config):
    for path in config.exp.paths.keys():
        if config.exp.paths[path] is not None and not isabs(config.exp.paths[path]):
            config.exp.paths[path]=join(get_original_cwd(),config.exp.paths[path])

@hydra.main(config_path='conf',config_name='tune')
def main(cfg:DictConfig):
    prefix_path(cfg)
    logger.info(f'{cfg}')
    device = cfg.exp.device
    if isdir(cfg.exp.paths.data_path):
        input_img_lst=[join(cfg.exp.paths.data_path,img) for img in natsorted(listdir(cfg.exp.paths.data_path))]
    else:
        input_img_lst = [cfg.exp.paths.data_path]
    if cfg.exp.max_samples is not None:
        input_img_lst=input_img_lst[:cfg.exp.max_samples]
    if cfg.exp.sample_lst is not None:
        input_img_lst=[input_img_lst[i] for i in cfg.exp.sample_lst]
    task=instantiate(cfg.exp.task,_convert_="all")
    regularization=instantiate(cfg.exp.regularization,_convert_="all")
    checkpoint = torch.load(cfg.exp.pretrained_checkpoint, map_location=device)
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if any(['network' in m for m in checkpoint.keys()]):
        new_checkpoint = {}
        for k,v in checkpoint.items():
            if 'network' in k:
                new_checkpoint[k[k.index('network')+8:]] = v
        checkpoint = new_checkpoint
    regularization.network.load_state_dict(checkpoint)
    restore=instantiate(cfg.exp.restore,_convert_="all",_partial_=True)(regularization=regularization,task=task)
    restore.to(device)
    
    for test_idx,test in enumerate(task.val_lst):
        aux=test.copy()
        del aux['name']
        logger.info(f'testing {test["name"]}')
        Path(join(test['name'],'images')).mkdir(parents=True,exist_ok=True)
        Path(join(test['name'],'curves')).mkdir(parents=True,exist_ok=True)
        if isinstance(task,super_resolution.SuperResolution) or isinstance(task,inpainting.Inpainting):
            aux['seed']=0
        avg_final_psnr=0

        for img_idx,img in enumerate(input_img_lst):
            logger.info(f'testing image {img_idx}')
            gt_img=to_tensor(open_img(img)).unsqueeze(0).to(device)
            if isinstance(task,mri.Mri):
                gt_img = (gt_img-gt_img.min())/(gt_img.max()-gt_img.min())
            if gt_img.shape[2]%2!=0 or gt_img.shape[3]%2!=0:
                h_pad=gt_img.shape[2]%2
                w_pad=gt_img.shape[3]%2
                gt_img=pad(gt_img,(0,w_pad,0,h_pad),mode='reflect')

            results=restore(gt_img,aux,extract_results=True)
            
            Path(join(test['name'],'images',f'{img_idx}')).mkdir(parents=True,exist_ok=True)

            imsave(join(test['name'],'images',f'{img_idx}','gt.png'),single2uint(tensor2single(gt_img)))
            imsave(join(test['name'],'images',f'{img_idx}','degraded.png'),single2uint(results['degraded_img']))
            imsave(join(test['name'],'images',f'{img_idx}','last_recon.png'),single2uint(results['recon_image']))
            imsave(join(test['name'],'images',f'{img_idx}','best_recon.png'),single2uint(results['peak_image']))
            avg_final_psnr+=results['final_psnr']
            if cfg.exp.output_every_step:
                Path(join(test['name'],'images',f'{img_idx}','every_step','x')).mkdir(parents=True,exist_ok=True)
                for k in range(len(results['x'])):
                    imsave(join(test['name'],'images',f'{img_idx}','every_step','x',f'x_{k}.png'),single2uint(results['x'][k]))
                Path(join(test['name'],'images',f'{img_idx}','every_step','z')).mkdir(parents=True,exist_ok=True)
                for k in range(len(results['z'])):
                    imsave(join(test['name'],'images',f'{img_idx}','every_step','z',f'z_{k}.png'),single2uint(results['z'][k]))
                Path(join(test['name'],'images',f'{img_idx}','every_step','Dx')).mkdir(parents=True,exist_ok=True)
                for k in range(len(results['Dx'])):
                    imsave(join(test['name'],'images',f'{img_idx}','every_step','Dx',f'Dx_{k}.png'),single2uint(results['Dx'][k]))
        avg_final_psnr/=len(input_img_lst)
        result_data_frame=pd.DataFrame({'final_psnr':[avg_final_psnr]})
        result_data_frame.to_csv(join(test['name'],'info.csv'),index=None)
if __name__ == '__main__':
    main()