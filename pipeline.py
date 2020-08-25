import os
from astropy.io import fits
from os.path import split, splitext
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        dataset_dir = os.path.join('./datasets', opt.dataset_name)
        self.input_format = opt.data_format_input
        self.target_format = opt.data_format_target

        if opt.is_train:
            self.label_path_list = sorted(glob(os.path.join(dataset_dir, 'Train', 'Input', '*.' + self.input_format)))
            self.target_path_list = sorted(glob(os.path.join(dataset_dir, 'Train', 'Target', '*.' + self.target_format)))

        else:
            self.label_path_list = sorted(glob(os.path.join(dataset_dir, 'Test', 'Input', '*.' + self.input_format)))
            

    def __getitem__(self, index):
        list_transforms = []
        list_transforms += []

        # [ Training data ] ==============================================================================================
        if self.opt.is_train:
            
            # [ Input ] ==================================================================================================
            if self.input_format in ["fits", "fts"]:                    
                IMG_A0 = np.array(fits.open(self.label_path_list[index])[0].data).transpose(2, 0 ,1)
            elif self.input_format in ["npy"]:
                IMG_A0 = np.load(self.label_path_list[index], allow_pickle=True)
            else:
                NotImplementedError("Please check data_format_input option. It has to be fits or npy.")
                
            IMG_A0[np.isnan(IMG_A0)] = 1
            UpIA = self.opt.saturation_upper_limit_input
            LoIA = self.opt.saturation_lower_limit_input
            
            label_array = np.log10(np.clip(IMG_A0, LoIA, UpIA))/(np.log10(UpIA) - np.log10(LoIA)) *2 - 1
            label_tensor = torch.tensor(label_array)
            
            
            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)
            
                
            # [ Target ] ==================================================================================================
            
            if self.target_format in ["fits", "fts"]:
                IMG_B0 = np.array(fits.open(self.target_path_list[index])[0].data)                
            elif self.input_format in ["npy"]:
                IMG_B0 = np.load(self.target_path_list[index], allow_pickle=True)
            else:
                NotImplementedError("Please check data_format_target option. It has to be fits or npy.")
                
            IMG_B0[np.isnan(IMG_A0)] = 1
            UpIB = self.opt.saturation_upper_limit_target
            LoIB = self.opt.saturation_lower_limit_target
            
            target_array = (np.clip(IMG_B0, LoIB, UpIB)-(UpIB+ LoIB))/((UpIB - LoIB)/2)
            target_tensor = torch.tensor(target_array, dtype=torch.float32)

            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor.unsqueeze(dim=0)  # Add channel dimension.
                
            return label_tensor, target_tensor, splitext(split(self.label_path_list[index])[-1])[0], \
                   splitext(split(self.target_path_list[index])[-1])[0]

        # [ Test data ] ===================================================================================================
        else:
            # [ Input ] ==================================================================================================
            if self.input_format in ["fits", "fts"]:                    
                IMG_A0 = np.array(fits.open(self.label_path_list[index])[0].data).transpose(2, 0 ,1)
            elif self.input_format in ["npy"]:
                IMG_A0 = np.load(self.label_path_list[index], allow_pickle=True)
            else:
                NotImplementedError("Please check data_format_input option. It has to be fits or npy.")
                
            IMG_A0[np.isnan(IMG_A0)] = 1
            UpIA = self.opt.saturation_upper_limit_input
            LoIA = self.opt.saturation_lower_limit_input
            
            label_array = np.log10(np.clip(IMG_A0, LoIA, UpIA))/(np.log10(UpIA) - np.log10(LoIA)) *2 - 1
            label_tensor = torch.tensor(label_array)
            
            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)
            
            return label_tensor, splitext(split(self.label_path_list[index])[-1])[0]
