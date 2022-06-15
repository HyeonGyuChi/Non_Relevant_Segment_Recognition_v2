import os
import re
import yaml
import json
import natsort
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

from core.util.database import DBHelper
from config.meta_db_config import subset_condition


class DBParser():
    # def __init__(self, args, state='train'):
    def __init__(self, args, state):
        self.args = args
        self.state = state
        self.data_dict = {}

        self.db_helper = DBHelper(args)
            
    def set_mini_fold(self):
        f_len = len(self.patients_list)
        N = self.args.n_mini_fold
        
        if f_len % N  != 0:
            raise 'T.T.....'
        
        mini_fold = int(self.args.train_stage[-1]) + 1
        
        split = f_len // N
        st = 0 + (mini_fold-1) * split
        ed = 0 + mini_fold * split
        
        if mini_fold == N:
            ed = f_len
            
        t_patients_list = []
                
        if self.state == 'train':                
            for target in range(f_len):
                if not (st <= target and target < ed):
                    t_patients_list.append(self.patients_list[target])                    
            
        elif self.state == 'val':
            for target in range(f_len):
                if (st <= target and target < ed):
                    t_patients_list.append(self.patients_list[target])
        
        self.patients_list = t_patients_list
            
    def get_patient_assets(self):
        patient_dict = {}
        
        for patient in self.data_dict.keys():
            p_dict = self.data_dict[patient]
            
            img_list, label_list = list(), list()
            for vd in p_dict.keys():
                img_list += list(p_dict[vd]['img'])
                
                if 'anno' in p_dict[vd]:
                    label_list += list(p_dict[vd]['anno'])
                else:
                    label_list = None
            
            if label_list is not None and len(label_list) != len(img_list):
                img_list = img_list[:len(label_list)]
            
            patient_dict[patient] = [img_list, label_list]
        
        return patient_dict
    
    def get_video_assets(self):
        video_dict = {}
        
        for patient in self.data_dict.keys():
            p_dict = self.data_dict[patient]
            video_dict[patient] = {}
            
            for vd in p_dict.keys():
                img_list = list(p_dict[vd]['img'])
                
                if 'anno' in p_dict[vd]:
                    label_list = list(p_dict[vd]['anno'])
                else:
                    label_list = None
                
                if label_list is not None and len(label_list) != len(img_list):
                    img_list = img_list[:len(label_list)]
                
                video_dict[patient][vd] = [img_list, label_list]
        
        return video_dict

    def load_data(self):
        asset_df = self.db_helper.select(subset_condition[self.state])
        self.patient_list = np.unique(asset_df['PATIENT'].values)

        # offline setup
        if 'mini' in self.args.train_stage:
            self.set_mini_fold()
            
        self.load_from_json(asset_df)

        
    def load_from_json(self, asset_df):
        for data in asset_df.values:
            # load frame list;..
            patient = data[2]

            if patient not in self.patient_list:
                continue

            # patient_path = self.args.data_base_path + '/toyset/{}/{}/{}'.format(*data[:3])
            patient_path = self.args.data_base_path + '/{}/{}/{}'.format(*data[:3])
            
            for video_name in natsort.natsorted(os.listdir(patient_path)):
                img_base_path = patient_path + f'/{video_name}/img'
                
                file_list = sorted(os.listdir(img_base_path))
            
                for fi, fname in enumerate(file_list):
                    file_list[fi] = img_base_path + f'/{fname}'

                if patient in self.data_dict:
                    self.data_dict[patient][video_name] = {'img': file_list}
                else:
                    self.data_dict[patient] = {
                        video_name: {'img': file_list}
                    }

                # make annotations
                anno_base_path = patient_path + f'/{video_name}/anno/org'

                # print("data[-1] ",data[-1] )
                # if data[-1] == True and os.path.exists(anno_base_path):
                if os.path.exists(anno_base_path):
                    anno_path = glob(anno_base_path + '/*.json')


                    # load annotationsubset_condition

                    with open(anno_path[0], 'r') as f:
                        anno_data = json.load(f)

                    # make annotation
                    labels = None #np.zeros(data['totalFrame'])
                    dup_cnt = 0
                    fix_list = []

                    for anno in anno_data['annotations']:
                        st, ed, label = anno['start'], anno['end'], anno['code']

                        if label != -1:
                            if labels is not None:
                                _labels = np.zeros(ed-st+1) + label
                                labels = np.concatenate((labels, _labels))
                            else:
                                labels = np.zeros(ed-st+1) + label

                            fix_list.append([st, ed])
                        else:
                            dup_cnt += 1
                        
                    if dup_cnt > 0:
                        file_list = []
                        t_file_list = self.data_dict[patient][video_name]['img']

                        for st, ed in fix_list:
                            file_list += t_file_list[st:ed+1]

                        # remove duplicate frames
                        self.data_dict[patient][video_name]['img'] = file_list

                    # quantization
                    labels = labels[::self.args.sample_ratio].astype('uint8')

                    self.data_dict[patient][video_name]['anno'] = labels
