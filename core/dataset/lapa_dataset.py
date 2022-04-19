import os
import sys
import random
import numpy as np
import torch
from glob import glob
from PIL import Image
import pandas as pd
import natsort
from torch.utils.data import Dataset

from core.dataset.aug_info import *
from core.util.parser import AssetParser
from core.util.sampler import BoundarySampler


class LapaDataset(Dataset):
    def __init__(self, args, state='train', sample_type='boundary'):
        super().__init__()
        
        self.args = args
        self.state = state
        self.sample_type = sample_type    
        self.img_list, self.label_list = None, None
        self.bs = BoundarySampler(self.args.IB_ratio, self.args.WS_ratio) 
        self.ap = AssetParser(self.args, state=self.state) 
        self.load_data()
         
        # augmentation setup
        if self.args.experiment_type == 'ours':
            if self.args.model == 'mobile_vit':
                d_transforms = data_transforms_mvit
            else:    
                d_transforms = data_transforms
        elif self.args.experiment_type == 'theator':
            d_transforms = data_transforms_theator
        
        self.aug = d_transforms[self.state]

    

    def __len__(self):
        return len(self.img_list)

    # return img, label
    def __getitem__(self, index):
        img_path, label = self.img_list[index], self.label_list[index]

        img = Image.open(img_path)
        img = self.aug(img)

        return img_path, img, label


    def load_data(self):
        self.ap.load_data()
        patient_data = self.ap.get_patient_assets()
        
        anno_df_list = []
       
        for patient, data in patient_data.items():
            anno_df = pd.DataFrame({
                'img_path': data[0],
                'class_idx': data[1],
            })
          
            if self.sample_type == 'boundary':
                # print('\n\n\t ==> HUERISTIC SAMPLING ... IB_RATIO: {}, WS_RATIO: {}\n\n'.format(self.args.IB_ratio, self.args.WS_ratio))
                anno_df['patient'] = anno_df.img_path.astype(str).str.split('/').str[7]

                anno_df = self.bs.sample(anno_df)[['img_path', 'class_idx']] 
     
            anno_df_list.append(anno_df)
        refine_df = pd.concat(anno_df_list)

        # hueristic_sampling
        if self.sample_type == 'boundary':
            assets_df = refine_df

        elif self.sample_type == 'random':
            print('\n\n\t ==> RANDOM SAMPLING ... IB_RATIO: {}\n\n'.format(self.args.IB_ratio))
            # random_sampling and setting IB:OOB data ratio
            # ratio로 구성 불가능 할 경우 전체 set 모두 사용
            nrs_ids = refine_df.index[refine_df['class_idx'] == 1].tolist()
            nrs_df = refine_df.loc[nrs_ids]
            rs_df = refine_df.drop(nrs_ids, inplace=True)
            
            max_ib_count, target_ib_count = len(rs_df), int(len(nrs_df)) * self.args.IB_ratio
            sampling_ib_count = max_ib_count if max_ib_count < target_ib_count else target_ib_count
            print('Random sampling from {} to {}'.format(max_ib_count, sampling_ib_count))
            
            rs_df = rs_df.sample(n=sampling_ib_count, replace=False) # 중복뽑기x, random seed 고정, OOB개수의 IB_ratio 개
            assets_df = pd.concat([rs_df, nrs_df])
            
        elif self.sample_type == 'all':
            print('\nSample type ALL')
            assets_df = refine_df
        else:
            raise 'Load Dataset Error'

        # last processing
        assets_df = assets_df[['img_path', 'class_idx']]
        self.img_list = assets_df.img_path.tolist()
        self.label_list = assets_df.class_idx.tolist()

        self.assets_df = assets_df






    def number_of_rs_nrs(self):
        return self.label_list.count(0) ,self.label_list.count(1)

    def number_of_patient_rs_nrs(self):
        patient_per_dic = {}

        val_assets_df = self.assets_df
        val_assets_df['patient'] = val_assets_df.img_path.str.split('/').str[6]
        patients_list = natsort.natsorted(list(set(val_assets_df['patient'])))
        
        total_rs_count, total_nrs_count = len(val_assets_df[val_assets_df['class_idx'] == 0]), \
                                            len(val_assets_df[val_assets_df['class_idx'] == 1])
            
        for patient in patients_list:
            patient_df = val_assets_df[val_assets_df['patient']==patient]
            patient_rs_count = len(patient_df[patient_df['class_idx'] == 0])
            patient_nrs_count = len(patient_df[patient_df['class_idx'] == 1])

            patient_per_dic.update(
                {
                    patient : {
                    'rs': patient_rs_count,
                    'nrs': patient_nrs_count,
                    'rs_ratio': patient_rs_count/total_rs_count,
                    'nrs_ratio': patient_nrs_count/total_nrs_count
                    } 
                }
            )

        return patient_per_dic