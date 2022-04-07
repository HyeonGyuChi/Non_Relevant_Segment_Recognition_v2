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

        # 작성중
        self.bs = BoundarySampler(self.args.IB_ratio, self.args.WS_ratio) 
        # 작성중
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
        # print("len(self.img_list)",len(self.img_list))
        return len(self.img_list)

    # return img, label
    def __getitem__(self, index):
        img_path, label = self.img_list[index], self.label_list[index]

        img = Image.open(img_path)
        img = self.aug(img)

        return img_path, img, label

    # 작성중
    def load_data(self):
        self.ap.load_data()
        # appointment_assets_path 존재하고  동시에 arg.train_stage 중 train/mini을 포함한 state라면 
        # -> load_data_from_path: appointment_assets_path의 csv 읽고 data_dict 설정

        # 아니면 
        # -> load_img_path_list: 
        # -> make_anno:

        patient_data = self.ap.get_patient_assets()
        # self.data_dict[patient].keys  중 self.data_dict[patient][keys]가 
        #img -> img data
        #anno 포함이면 -> label_list에 추가
        #anno 미포함이면 -> label_list = None

        #-> label_list None이 아니고 동시에 img-label 개수가 안 맞으면 img_list = img_list[:len(label_list)]
        # patient_dict[patient] = [img_list, label_list]
        # return patient_dict

        anno_df_list = []
        # print("load_data_patient_data",patient_data)
        # print("load_data_patient_data.items()",patient_data.items())

        for patient, data in patient_data.items():
            anno_df = pd.DataFrame({
                'img_path': data[0],
                'class_idx': data[1],
            })
            
            
            # print("patient",patient)
            # print("anno_df\n",anno_df)

            if self.sample_type == 'boundary':
                # print('\n\n\t ==> HUERISTIC SAMPLING ... IB_RATIO: {}, WS_RATIO: {}\n\n'.format(self.args.IB_ratio, self.args.WS_ratio))
                anno_df['patient'] = anno_df.img_path.astype(str).str.split('/').str[7]

                anno_df = self.bs.sample(anno_df)[['img_path', 'class_idx']] 
                #print("second anno_df\n",anno_df)
            
            anno_df_list.append(anno_df)
            # print("anno_df_list\n",anno_df_list)
            # print(patient, '   end')

        
        refine_df = pd.concat(anno_df_list)
        # print("refine_df.head()",refine_df.head())

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

     # 해당 robot_dataset의 patinets별 assets 개수 
     # (hem train할때 valset만들어서 hem_helper의 args.hem_per_patinets에서 사용됨.)     
     # 작성중
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