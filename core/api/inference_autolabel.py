import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.model import get_model
from core.dataset import SubDataset
from core.dataset import InferDataset
from core.util.parser import AssetParser
from core.util.parser import DBParser
# from core.util.sampler import IntervalSampler
from core.util.misc import *
from core.util.anno2json import Anno2Json


class InferenceDB_autolabel():
    def __init__(self, args):
        self.args = args
        self.ap = AssetParser(self.args, state='val')
        self.dp = DBParser(self.args, state='test')
        self.dset = InferDataset(self.args)

    def load_model(self):
        model_path = get_inference_model_path(self.args.restore_path, load_type='best')
        ckpt_state = torch.load(model_path) 
        self.model = get_model(self.args).to(self.args.device)
        self.model.load_state_dict(ckpt_state['model'])
        
    def set_inference_interval(self, inference_interval):
        self.inference_interval = inference_interval
        
    def load_dataset(self):
        self.ap.load_data()
        self.video_assets = self.ap.get_video_assets()

    def load_dataset_autolabel(self):
        self.dp.load_data()
        self.video_assets = self.dp.get_video_assets()
    
    def find_patient_no(self, video_name):
        tokens = video_name.split('_')
        # search patient number
        for ti, token in enumerate(tokens):
            global patient_num
            if self.args.dataset == 'robot':
                if token == 'R':
                    patient_num = 'R_' + tokens[ti+1]
                    # print("patient_num",patient_num)
                    break
            elif self.args.dataset == 'lapa':
                if token == 'L':
                    #patient = 'L_' + tokens[ti+1]
                    patient_num = video_name.split("_")[0]+"_"+video_name.split("_")[1]+"_"+video_name.split("_")[2]+"_"+video_name.split("_")[3]+"_"+video_name.split("_")[4]
                    # print("patient_num",patient_num)
                    break

        return patient_num
    
    @torch.no_grad()
    def inference(self):
        import numpy as np
        print('\n\t########## INFERENCEING (DB) #########\n')
        
        # data loder
        self.load_dataset()
        results = {}
        results_want={}
        
        for patient_n in self.video_assets.keys():
            patient_data = self.video_assets[patient_n]
            results[patient_n] = {}
            results_want[patient_n]={}
            
            for video_name in patient_data.keys():
                patient = self.find_patient_no(video_name)
                each_patients_save_dir = self.args.save_path + '/inference_results/{}'.format(patient)
                
                data = patient_data[video_name]
                
                self.dset.set_img_list(data[0])
                dl = DataLoader(dataset=self.dset,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.num_workers*3,
                                pin_memory=True,
                                )
            
                # inference log
                predict_list = []
                target_img_list = []
                target_frame_idx_list = []
                
                print('expected loop-cnt : {:.1f}'.format(len(dl) / self.inference_interval))

                # inferencing model
                for sample in tqdm(dl, desc='Inferencing... \t ==> {}'.format(video_name)) :
                    batch_input = sample['img'].cuda()
                    batch_output = self.forward(batch_input)

                    # predict
                    batch_predict = torch.argmax(batch_output.cpu(), 1)
                    batch_predict = batch_predict.tolist()

                    # save results
                    predict_list += list(batch_predict)
                    target_img_list += sample['img_path'] # target img path
                    target_frame_idx_list += sample['db_idx'] # target DB

                try:
                    target_frame_idx_list = list(map(int, target_frame_idx_list)) 
                    # '0000000001' -> 1
                except:
                    target_frame_idx_list_new=[]
                    for i in range(len(target_frame_idx_list)):
                        if "-" in target_frame_idx_list[i]:
                            target_frame_idx = target_frame_idx_list[i].split("-")[-1]
                            target_frame_idx_list_new.append(target_frame_idx)
                        else:
                            target_frame_idx_list_new.append(target_frame_idx_list[i])
                            
                gt_list = data[1]
                if gt_list is None:
                    gt_list = list(np.zeros(len(predict_list))-1)
                
                predict_df = pd.DataFrame({
                            'frame_idx': target_frame_idx_list,
                            'predict': predict_list,
                            'gt': gt_list,
                            'target_img': target_img_list,
                        })
                
                self.save_results(predict_df, each_patients_save_dir, video_name)
                results[patient][video_name] = predict_df

            
        return results

    @torch.no_grad()
    def inference_autolabel(self):
        import numpy as np
        import json
        print('\n\t########## INFERENCEING (DB) #########\n')
        
        # data loder
        self.load_dataset_autolabel()
        results = {}
        results_want={}
        
        for patient_n in self.video_assets.keys():
            patient_data = self.video_assets[patient_n]
            results[patient_n] = {}
            results_want[patient_n]={}
            
            for video_name in patient_data.keys():
                # patient = self.find_patient_no(video_name)
                each_patients_save_dir = self.args.save_path + '/inference_results/{}'.format(patient_n)
                
                data = patient_data[video_name]
                
                self.dset.set_img_list(data[0])
                dl = DataLoader(dataset=self.dset,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.num_workers*3,
                                pin_memory=True,
                                )
            
                # inference log
                predict_list = []
                target_img_list = []
                target_frame_idx_list = []
                
                print('expected loop-cnt : {:.1f}'.format(len(dl) / self.inference_interval))

                # inferencing model
                for sample in tqdm(dl, desc='Inferencing... \t ==> {}'.format(video_name)) :
                    batch_input = sample['img'].cuda()
                    batch_output = self.forward(batch_input)

                    # predict
                    batch_predict = torch.argmax(batch_output.cpu(), 1)
                    batch_predict = batch_predict.tolist()

                    # save results
                    predict_list += list(batch_predict)
                    target_img_list += sample['img_path'] # target img path
                    target_frame_idx_list += sample['db_idx'] # target DB

                try:
                    target_frame_idx_list = list(map(int, target_frame_idx_list)) 
                    # '0000000001' -> 1
                except:
                    target_frame_idx_list_new=[]
                    for i in range(len(target_frame_idx_list)):
                        if "-" in target_frame_idx_list[i]:
                            target_frame_idx = target_frame_idx_list[i].split("-")[-1]
                            target_frame_idx_list_new.append(target_frame_idx)
                        else:
                            target_frame_idx_list_new.append(target_frame_idx_list[i])
                            
                gt_list = data[1]
                if gt_list is None:
                    gt_list = list(np.zeros(len(predict_list))-1)
                
                predict_df = pd.DataFrame({
                            'frame_idx': target_frame_idx_list,
                            'predict': predict_list,
                            'gt': gt_list,
                            'target_img': target_img_list,
                        })
                
                self.save_results(predict_df, each_patients_save_dir, video_name)
                results[patient_n][video_name] = predict_df

                results_target_img_list = results[patient_n][video_name]["target_img"].tolist()
                results_predict_list = results[patient_n][video_name]["predict"].tolist()
                results[patient_n][video_name]=[results_target_img_list,results_predict_list]
        
        annotation_to_json = Anno2Json(self.args,results,None)
        annotation_to_json.make_json(version="autolabel")

    
    def forward(self, batch_input):
        if 'online' in  self.args.hem_extract_mode:
            _, batch_output = self.model(batch_input)
        else:
            batch_output = self.model(batch_input)
            
        return batch_output

    
    def save_results(self, predict_df, each_patients_save_dir, video_name):
        # for save video results
        each_videos_save_dir = os.path.join(each_patients_save_dir, video_name)
        os.makedirs(each_videos_save_dir, exist_ok=True)
        
        predict_csv_path = os.path.join(each_videos_save_dir, '{}.csv'.format(video_name))
        predict_df.to_csv(predict_csv_path)


