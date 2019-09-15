## Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_test
from PIL import Image
import torch
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        if opt.isTrain or opt.use_encoded_image:
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            self.AR_paths = make_dataset(self.dir_A)

        ### input A inter 1 (label maps)
        if opt.isTrain or opt.use_encoded_image:
            dir_A_inter_1 = '_label_inter_1'
            self.dir_A_inter_1 = os.path.join(opt.dataroot, opt.phase + dir_A_inter_1)
            self.A_paths_inter_1 = sorted(make_dataset(self.dir_A_inter_1))

        ### input A inter 2 (label maps)
        if opt.isTrain or opt.use_encoded_image:
            dir_A_inter_2 = '_label_inter_2'
            self.dir_A_inter_2 = os.path.join(opt.dataroot, opt.phase + dir_A_inter_2)
            self.A_paths_inter_2 = sorted(make_dataset(self.dir_A_inter_2))

        ### input A test (label maps)
        if not (opt.isTrain or opt.use_encoded_image):
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset_test(self.dir_A))
            dir_AR = '_AR' if self.opt.label_nc == 0 else '_labelref'
            self.dir_AR = os.path.join(opt.dataroot, opt.phase + dir_AR)
            self.AR_paths = sorted(make_dataset_test(self.dir_AR))

        ### input B (real images)
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.BR_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths) 
 
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        AR_path = self.AR_paths[index]
        A = Image.open(A_path)
        AR = Image.open(AR_path)

        if self.opt.isTrain:
            A_path_inter_1 = self.A_paths_inter_1[index]
            A_path_inter_2 = self.A_paths_inter_2[index]
            A_inter_1 = Image.open(A_path_inter_1)
            A_inter_2 = Image.open(A_path_inter_2)
  
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
            if self.opt.isTrain:
                A_inter_1_tensor = transform_A(A_inter_1.convert('RGB'))
                A_inter_2_tensor = transform_A(A_inter_2.convert('RGB'))
            AR_tensor = transform_A(AR.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0
            if self.opt.isTrain:
                A_inter_1_tensor = transform_A(A_inter_1) * 255.0
                A_inter_2_tensor = transform_A(A_inter_2) * 255.0
            AR_tensor = transform_A(AR) * 255.0
        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        B_path = self.B_paths[index]   
        BR_path = self.BR_paths[index]   
        B = Image.open(B_path).convert('RGB')
        BR = Image.open(BR_path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)
        BR_tensor = transform_B(BR)

        if self.opt.isTrain:
            input_dict = {'inter_label_1': A_inter_1_tensor, 'label': A_tensor, 'inter_label_2': A_inter_2_tensor, 'label_ref': AR_tensor, 'image': B_tensor, 'image_ref': BR_tensor, 'path': A_path, 'path_ref': AR_path}
        else:
            input_dict = {'label': A_tensor, 'label_ref': AR_tensor, 'image': B_tensor, 'image_ref': BR_tensor, 'path': A_path, 'path_ref': AR_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
