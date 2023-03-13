import numpy as np
from typing import Dict, Optional, Sequence, Tuple, Union
import math
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import scipy.io as sio
# from ismrmrdtools import show, transform
# import ReadWrapper
from torch.utils.data.dataset import Dataset
from torch.nn import init

import os
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import scipy.io as sp
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute
from models import networks

def make_vdrs_mask(N1,N2,nlines,init_lines):
    # Setting Variable Density Random Sampling (VDRS) mask
    mask_vdrs=np.zeros((N1,N2),dtype='bool')
    mask_vdrs[:,int(0.5*(N2-init_lines)):int(0.5*(N2+init_lines))]=True
    nlinesout=int(0.5*(nlines-init_lines))
    rng = np.random.default_rng()
    t1 = rng.choice(int(0.5*(N2-init_lines))-1, size=nlinesout, replace=False)
    t2 = rng.choice(np.arange(int(0.5*(N2+init_lines))+1, N2), size=nlinesout, replace=False)
    mask_vdrs[:,t1]=True; mask_vdrs[:,t2]=True
    return mask_vdrs
# mask = make_vdrs_mask(nx,ny,np.int(ny*0.25),np.int(ny*0.08)) # 4x
# mask = make_vdrs_mask(nx,ny,np.int(ny*0.125),np.int(ny*0.04)) # 8x

def loadData(Kspace_data_name, mask_data_name, num_train, num_test, batch_size, targeted_acceleration):
    kspace_array = os.listdir(Kspace_data_name)
    kspace_array = sorted(kspace_array)

    image_space_data = []

    ## Loading the kspace data of size (sentive_map_real(coil,channel,size,size),sentive_map_img(coil,channel,size,size), kspace_real(coil,channel,size,size),kspace_img(coil,channel,size,size))
    # print("loading sensitive map")
    for j in range(len(kspace_array)):
        if len(image_space_data) > num_train + num_test:
            break
        kspace_file = kspace_array[j]
        kspace_data_from_file = np.load(os.path.join(Kspace_data_name, kspace_file), 'r')
        if kspace_data_from_file['k_r'].shape[2] < 373 and kspace_data_from_file['k_r'].shape[2] > 367:
            image_space_data.append(kspace_data_from_file)
    # print("finish loading sensitive map")
    mask_array = os.listdir(mask_data_name)
    mask_array = sorted(mask_array)
    mask_file = mask_array[0] # mask shape would be 640 368
    mask_from_file = np.load(os.path.join(mask_data_name, mask_file), 'r')

    nx, ny = 640, 368
    tol = 0.05
    alpha = 0.2
    orig_acceleration = len(mask_from_file[0]) / np.sum(mask_from_file[0])
    acceleration = targeted_acceleration

    mask_real = np.ones((nx, ny), dtype='bool')

    for _ in range(20): 
        if len(mask_real[0]) / np.sum(mask_real[0]) > (1 + tol) * targeted_acceleration:
            acceleration *= 1 - alpha
        elif len(mask_real[0]) / np.sum(mask_real[0]) < (1 - tol) * targeted_acceleration:
            acceleration *= 1 + alpha
        else:
            break
        mask_real = make_vdrs_mask(nx, ny, np.int(ny / acceleration), np.int(ny * 0.32 / acceleration))
        if targeted_acceleration > orig_acceleration:
            mask_real = np.logical_and(mask_real, mask_from_file)
        else:
            mask_real = np.logical_or(mask_real, mask_from_file)

    real_acceleration = len(mask_real[0]) / np.sum(mask_real[0])
    print(f'real acceleration: {real_acceleration:.4f}')
    
    mask_data_select = []
    mask_data_real = []
    for e in range(len(image_space_data)):
        if image_space_data[e]['k_r'].shape[2] > 368:
            mask_data_select.append(np.pad(mask_from_file, ((0, 0), (2, 2)), 'constant'))
            mask_data_real.append(np.pad(mask_real, ((0, 0), (2, 2)), 'constant'))
        else:
            mask_data_select.append(mask_from_file)
            mask_data_real.append(mask_real)
    # print("finish loading mask")

    test_clean_paths = image_space_data[num_train:num_train + num_test]
    mask_test_paths = mask_data_select[num_train:num_train + num_test]
    mask_real_paths_test = mask_data_real[num_train:num_train + num_test]
    test_dataset = nyumultidataset(test_clean_paths, mask_test_paths, mask_real_paths_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return real_acceleration, None, test_loader


class nyumultidataset(Dataset): # model data loader
    def  __init__(self, kspace_data, mask_data, mask_data_real):
        self.A_paths = kspace_data
        # self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.mask_path = mask_data
        # self.mask_path =sorted(self.mask_path)
        self.mask_data_real = mask_data_real
        self.nx = 640
        self.ny = 368

    def __getitem__(self, index):
        A_temp = self.A_paths[index]
        s_r = A_temp['s_r']/ 32767.0 
        s_i = A_temp['s_i']/ 32767.0 
        k_r = A_temp['k_r']/ 32767.0
        k_i = A_temp['k_i']/ 32767.0 
        _, nx, ny = s_r.shape
        mask = self.mask_path[index]
        mask_real = self.mask_data_real[index]
        k_np = np.stack((k_r, k_i), axis=0)
        s_np = np.stack((s_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                         s_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)
        mask = torch.tensor(np.repeat(mask[np.newaxis, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160], 2, axis=0), dtype=torch.float32)
        mask_real = torch.tensor(np.repeat(mask_real[np.newaxis, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160], 2, axis=0), dtype=torch.float32)
        A_k = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3)
        A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A_I = A_I[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        A_s = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)),dim=0)
        A_I = A_I/torch.max(torch.abs(SOS)[:])
        A_k = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2)
        kreal = A_k
        AT = networks.OPAT2(A_s)
        # Iunder = AT(kreal, mask)
        Iunder = AT(kreal, mask_real)
        Ireal = AT(kreal, torch.ones_like(mask))
        return Iunder, Ireal, A_s, mask
     
       
    def __len__(self):
        return len(self.A_paths)
