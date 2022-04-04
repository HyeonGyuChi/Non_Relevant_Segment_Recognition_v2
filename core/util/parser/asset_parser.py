import os
import re
import yaml
import json
import natsort
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob


from core.dataset.fold_robot import *
# from core.dataset.fold_lapa import *



class AssetParser():
    def __init__(self, args, state='train'):
        self.args = args
        self.state = state
        self.data_dict = {}
    
        if 'mini' in self.args.train_stage:
            self.set_asset_info(state='train')
            self.set_mini_fold()
        else:
            self.set_asset_info(state=self.state)
                   
    def set_asset_info(self, state='train'):
        if self.args.dataset == 'robot':
            if self.args.datatype == 'vihub':
                pass
            elif self.args.datatype == 'etc':
                pass
            elif self.args.datatype == 'mola':
                self.patients_list = mola[state][self.args.fold]
        elif self.args.dataset == 'lapa':
            pass
        
        self.img_base_path = self.args.data_base_path + '/{}/{}/img'.format(self.args.dataset,
                                                                                self.args.datatype)
        self.anno_path = self.args.data_base_path + '/{}/{}/anno/{}'.format(self.args.dataset,
                                                                                self.args.datatype,
                                                                                self.args.data_version)
            
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
        if self.args.appointment_assets_path != '' and \
            (self.state == 'train' or 'mini' in self.args.train_stage):
            self.load_data_from_path()
        else:
            self.load_img_path_list()
            self.make_anno()
        
    def load_data_from_path(self):
        appointment_assets_df = pd.read_csv(self.args.appointment_assets_path)
        
        # reset data_dict
        self.data_dict = {}
        
        asset_info = appointment_assets_df.values
        path_list = appointment_assets_df.img_path.str.split('/').str[6]
                
        for vals in asset_info:
            _, img_path, label, _ = vals
            
            tokens = img_path.split('/')
            patient_no, video_name = tokens[6], tokens[7]
                        
            if patient_no in self.patients_list:
                if patient_no in self.data_dict:
                    if video_name in self.data_dict[patient_no]: 
                        self.data_dict[patient_no][video_name]['img'].append(img_path)
                        self.data_dict[patient_no][video_name]['anno'].append(label)
                    else:
                        self.data_dict[patient_no][video_name] = {}
                        self.data_dict[patient_no][video_name]['img'] = [img_path]
                        self.data_dict[patient_no][video_name]['anno'] = [label]
                else:
                    self.data_dict[patient_no] = {}
                    self.data_dict[patient_no][video_name] = {}
                    self.data_dict[patient_no][video_name]['img'] = [img_path]
                    self.data_dict[patient_no][video_name]['anno'] = [label]
                 
    def load_img_path_list_only(self):
        patients_list = natsort.natsorted(os.listdir(self.img_base_path))
        
        for patient in tqdm(patients_list):
            p_path = self.img_base_path + f'/{patient}'
            video_list = natsort.natsorted(os.listdir(p_path))
            
            for video_name in video_list:
                v_path = p_path + f'/{video_name}'
                
                file_list = sorted(os.listdir(v_path))
                
                for fi, fname in enumerate(file_list):
                    file_list[fi] = v_path + f'/{fname}'
                
                if self.args.sample_ratio > 1:
                    file_list = file_list[::self.args.sample_ratio]
                
                if patient in self.data_dict:
                    self.data_dict[patient][video_name] = {'img': file_list}
                else:
                    self.data_dict[patient] = {
                        video_name: {'img': file_list}
                    }
        
    def load_img_path_list(self):
        for patient in tqdm(self.patients_list):
            p_path = self.img_base_path + f'/{patient}'
            video_list = natsort.natsorted(os.listdir(p_path))
            
            for video_name in video_list:
                v_path = p_path + f'/{video_name}'
                
                file_list = sorted(os.listdir(v_path))
                
                for fi, fname in enumerate(file_list):
                    file_list[fi] = v_path + f'/{fname}'
                
                if self.args.sample_ratio > 1:
                    file_list = file_list[::self.args.sample_ratio]
                
                if patient in self.data_dict:
                    self.data_dict[patient][video_name] = {'img': file_list}
                else:
                    self.data_dict[patient] = {
                        video_name: {'img': file_list}
                    }

    def make_anno(self):
        anno_list = natsort.natsorted(glob(self.anno_path + '/*.json'))
        
        for anno_path in tqdm(anno_list):
            anno_fname = anno_path.split('/')[-1][:-5]
            tokens = anno_fname.split('_')
            
            # search patient number
            for ti, token in enumerate(tokens):
                if self.args.dataset == 'robot':
                    if token == 'R':
                        patient = 'R_' + tokens[ti+1]
                        break
                elif self.args.dataset == 'lapa':
                    if token == 'L':
                        patient = 'L_' + tokens[ti+1]
                        break

            if patient in self.data_dict:
                # load annotation
                with open(anno_path, 'r') as f:
                    data = json.load(f)

                # make annotation
                labels = np.zeros(data['totalFrame'])
                
                for anno in data['annotations']:
                    st, ed = anno['start'], anno['end']
                    labels[st:ed+1] = 1
                    
                # quantization
                labels = labels[::self.args.sample_ratio].astype('uint8')
    
                for video_name in self.data_dict[patient].keys():
                    if video_name in anno_fname:
                        self.data_dict[patient][video_name]['anno'] = labels