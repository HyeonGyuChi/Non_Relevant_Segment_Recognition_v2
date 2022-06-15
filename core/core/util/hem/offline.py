import sys
import os, json, natsort
import pickle
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.model import get_model
from core.dataset import load_data
from core.util.misc import get_inference_model_path


class OfflineHEM():
    def __init__(self, args):
        self.args = args
        
        self.n_classes = 2
        self.NON_HEM, self.HEM = (0, 1)
        self.IB_CLASS, self.OOB_CLASS = (0, 1)
        
        self.hem_final_df_columns = ['img_path', 'class_idx', 'HEM']
        self.method_to_func = {
            'hem-softmax_diff_small-offline': ['extract_hem_from_softmax_diff', 'small'],
            'hem-softmax_diff_large-offline': ['extract_hem_from_softmax_diff', 'large'],
            'hem-voting-offline': ['extract_hem_from_voting', None],
            'hem-mi_small-offline': ['extract_hem_from_mutual_info', 'small'],
            'hem-mi_large-offline': ['extract_hem_from_mutual_info', 'large'],
        }
        
        self.softmax = nn.Softmax(dim=1)
        
    def load_data(self, save_path):
        tokens = save_path.split('/')[-1].split('_')
        
        for ti, token in enumerate(tokens):
            if token == 'stage':
                mini_fold = tokens[ti+1].split('-')[0]
        
        args = copy.deepcopy(self.args)
        args.train_stage = 'mini_fold_stage_{}'.format(mini_fold)
        
        _, val_loader = load_data(args)
        self.dset = val_loader.dataset
        self.target_patient_dict = self.dset.number_of_patient_rs_nrs()
        
    def load_model(self, model_path):
        ckpt_state = torch.load(model_path)
        
        self.model = get_model(self.args).to(self.args.device)
        self.model.load_state_dict(ckpt_state['model'])
        
    def get_path_list(self):
        N = self.args.n_mini_fold
        stage = self.args.cur_stage
        
        save_dir_list = natsort.natsorted(glob(self.args.save_path + '/*mini_fold*'))
        
        save_dir_list = [
            save_path.split('/')[-1] for save_path in save_dir_list
        ]
        
        l_list = len(save_dir_list)
        
        if l_list // N >= stage:
            st, ed = N * (stage-1), N * stage
            dir_list = save_dir_list[st:ed]
            
            save_path_list = [self.args.save_path + '/{}'.format(dir_name) for dir_name in dir_list]
            
            ckpt_path_list = [get_inference_model_path(self.args.save_path + '/{}'.format(dir_name)) \
                                for dir_name in dir_list]
        else:
            raise 'Not found mini fold checkpoints'
        
        return save_path_list, ckpt_path_list
        
    def extract(self, method_name='hem-softmax_diff_small-offline'):
        save_path_list, ckpt_path_list = self.get_path_list()
        
        for save_path, ckpt_path in zip(save_path_list, ckpt_path_list):
            self.load_model(ckpt_path)
            self.load_data(save_path)
            
            dropout_predictions, gt_list, img_path_list = self.compute_mcd()
            
            if method_name == 'all': # extract hem assets for all methods
                for method_name in self.method_to_func.keys():
                    hem_final_df = self.apply_method(dropout_predictions, gt_list, img_path_list, method_name)
                    
                    hem_final_df_path = os.path.join(save_path, '{}.csv'.format(method_name))
                    hem_final_df.to_csv(hem_final_df_path, index=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)
            else:
                hem_final_df = self.apply_method(dropout_predictions, gt_list, img_path_list, method_name)
                
                hem_final_df_path = os.path.join(save_path, '{}.csv'.format(method_name))
                hem_final_df.to_csv(hem_final_df_path, index=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)
            
    def aggregate_assets(self, method_name='all'):
        save_path_list, _ = self.get_path_list()
        
        all_hem_assets_dir = os.path.join(save_path_list[0], 'hem_assets')
        os.makedirs(all_hem_assets_dir)
        
        if method_name == 'all': # aggreagte hem assets for all methods
            for method_name in self.method_to_func.keys():
                agg_hem_assets_df = pd.DataFrame([]) # init
                
                for save_path in save_path_list:
                    hem_path = save_path + '/{}.csv'.format(method_name)
                    hem_df = pd.read_csv(hem_path)
                    agg_hem_assets_df = agg_hem_assets_df.append(hem_df, ignore_index=True)
                    
                agg_hem_assets_path = all_hem_assets_dir + '/{}-agg.csv'.format(method_name)
                agg_hem_assets_df.to_csv(agg_hem_assets_path)
            
        else:
            agg_hem_assets_df = pd.DataFrame([]) # init
                
            for save_path in save_path_list:
                hem_path = save_path + '/{}.csv'.format(method_name)
                hem_df = pd.read_csv(hem_path)
                agg_hem_assets_df = agg_hem_assets_df.append(hem_df, ignore_index=True)
                
            agg_hem_assets_path = all_hem_assets_dir + '/{}-agg.csv'.format(method_name)
            agg_hem_assets_df.to_csv(agg_hem_assets_path)
    
    def get_aggregate_path(self, method_name='hem-softmax_diff_small-offline'):
        save_path_list, _ = self.get_path_list()
        
        all_hem_assets_dir = os.path.join(save_path_list[0], 'hem_assets')
        agg_hem_assets_path = all_hem_assets_dir + '/{}-agg.csv'.format(method_name)
        
        return agg_hem_assets_path
        
    
    def apply_method(self, dropout_predictions, gt_list, img_path_list, method_name):
        # hem_final_df 초기화
        hem_final_df = pd.DataFrame(columns=self.hem_final_df_columns)

        # extracting hem (patient 별 divide)
        assets_df = pd.DataFrame(img_path_list, columns=['img_path'])
        assets_df['patient'] = assets_df.img_path.str.split('/').str[6] # patient 파싱
        assets_df['gt'] = gt_list

        patients_list = natsort.natsorted(list(set(assets_df['patient'])))

        for patient in tqdm(patients_list, desc='Extract HEM Assets per patients ...'):
            print('Patinet : {}'.format(patient))

            patient_idx = assets_df.index[assets_df['patient'] == patient].tolist()
            patient_df = assets_df.loc[patient_idx]
            
            patient_img_path_list, patient_gt_list = patient_df['img_path'].tolist(), patient_df['gt'].tolist()
            patient_dropout_predictions = dropout_predictions[:, patient_idx, :] # patient_dropout_predictions.shape = (5, n_patient, 2)
            
            print('\ngenerate hem mode : {}\n'.format(method_name))

            method_info = self.method_to_func[method_name]

            hem_method = getattr(self, method_info[0]) # func
            data = hem_method(patient_dropout_predictions, 
                                patient_gt_list, 
                                patient_img_path_list,
                                method_info[1])
            
            patient_hem_final_df = self.set_ratio(*data, patient)
            hem_final_df = hem_final_df.append(patient_hem_final_df, ignore_index=True) # hem_final_df 에 patients 별 결과 추가

        return hem_final_df
    
    @torch.no_grad()
    def compute_mcd(self):
        # init for parameter for hem methods
        img_path_list = []
        gt_list = []

        d_loader = DataLoader( # validation
            self.dset,
            batch_size=self.args.batch_size, # self.args.batch_size로 사용할수 있지만 args 배재
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers * 3, # self.args.num_workers
            pin_memory=True,
        )
        
        img_path_list = self.dset.img_list
        gt_list = self.dset.label_list
        
        ### 0. MC paramter setting
        
        n_samples = len(img_path_list)
        dropout_predictions = np.empty((0, n_samples, self.n_classes)) 
        self.model.eval()

        ## 1. MC forward
        for cnt in tqdm(range(self.args.n_dropout), desc='MC FORWARDING ... '):
            predictions = np.empty((0, self.n_classes))
            
            print('{}th MC FORWARDING ...'.format(cnt+1))
            
            for data in tqdm(d_loader, desc='processing...'):
                y_hat = self.model(data[1].cuda())
                y_hat = self.softmax(y_hat)

                predictions = np.vstack((predictions, y_hat.cpu().numpy()))

            # dropout predictions - shape (forward_passes, n_samples, n_classes)
            dropout_predictions = np.vstack((dropout_predictions,
                                        predictions[np.newaxis, :, :]))  
    
        return dropout_predictions, gt_list, img_path_list
    
    def enable_dropout(self):
        dropout_layer = []
        
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                dropout_layer.append(m)
        
        if self.args.n_dropout > 1 :
            dropout_layer[-1].train() # only last layer to train
    
    def get_answer(self, dropout_predictions, gt_list, answer_type='mean'):
        if answer_type == 'mean':
            mean = np.mean(dropout_predictions, axis=-1)
            predict_np = np.argmax(mean, axis=0)
            
            answer = predict_np == np.array(gt_list) # compare with gt list
        else:
            predict_table = np.argmax(dropout_predictions, axis=2) # (forward_passes, n_samples)
            predict_ratio = np.mean(predict_table, axis=0) # (n_samples)

            predict_np = np.around(predict_ratio) # threshold == 0.5, if predict_ratio >= 0.5, predict_class == OOB(1)
            predict_np = np.int8(predict_np) # casting float to int
            predict_list = predict_np.tolist() # to list

            answer = predict_np == np.array(gt_list) # compare with gt list

        return answer, predict_np
    
    # TODO 나중에 수정
    def set_ratio(self, 
                  hard_neg_df, hard_pos_df, 
                  vanila_neg_df, vanila_pos_df, 
                  patient_no=None):
        
        patient_nrs_count = int(self.target_patient_dict[patient_no]['nrs'])
        patient_rs_count = int(patient_nrs_count * self.args.IB_ratio)
        target_rs_cnt, target_nrs_cnt = patient_nrs_count, patient_rs_count
        
        # df 정렬
        hard_neg_df = hard_neg_df.sort_values(by='Img_path')
        hard_pos_df = hard_pos_df.sort_values(by='Img_path')
        vanila_neg_df = vanila_neg_df.sort_values(by='Img_path')
        vanila_pos_df = vanila_pos_df.sort_values(by='Img_path')

        # HEM 표기
        hard_neg_df['HEM'] = [1]*len(hard_neg_df)
        hard_pos_df['HEM'] = [1]*len(hard_pos_df)
        vanila_neg_df['HEM'] = [0]*len(vanila_neg_df)
        vanila_pos_df['HEM'] = [0]*len(vanila_pos_df)


        # train data 수 만큼 hem data extract
        if len(hard_pos_df) > target_nrs_cnt: hard_pos_df = hard_pos_df.sample(n=target_nrs_cnt, replace=False)
        if len(hard_neg_df) > target_rs_cnt: hard_neg_df = hard_neg_df.sample(n=target_rs_cnt, replace=False) 

        target_len_vanila_pos = target_nrs_cnt - len(hard_pos_df)
        target_len_vanila_neg = target_rs_cnt - len(hard_neg_df)

        
        try:
            vanila_pos_df = vanila_pos_df.sample(n=target_len_vanila_pos, replace=False) # 중복뽑기x, random seed 고정, hem_oob 개
        except:
            vanila_pos_df = vanila_pos_df.sample(frac=1, replace=False) # 중복뽑기x, random seed 고정, 전체 oob_df

        try:
            vanila_neg_df = vanila_neg_df.sample(n=target_len_vanila_neg, replace=False) # 중복뽑기x, random seed 고정, target_ib_assets_df_len 개
        except:
            vanila_neg_df = vanila_neg_df.sample(frac=1, replace=False)

        final_pos_assets_df = pd.concat([hard_pos_df, vanila_pos_df])[['Img_path', 'GT', 'HEM']]
        final_neg_assets_df = pd.concat([hard_neg_df, vanila_neg_df])[['Img_path', 'GT', 'HEM']]

        # sort & shuffle
        final_assets_df = pd.concat([final_pos_assets_df, final_neg_assets_df]).sort_values(by='Img_path', axis=0, ignore_index=True)
        print('\tSORT final_assets HEAD\n', final_assets_df.head(20), '\n\n')

        final_assets_df = final_assets_df.sample(frac=1).reset_index(drop=True)
        print('\tSHUFFLE final_assets HEAD\n', final_assets_df.head(20), '\n\n')
        
        final_assets_df.columns = ['img_path', 'class_idx', 'HEM']

        return final_assets_df
    
    def split_to_hem_vanila_df(self, hard_df, vanila_df): # total_df should have ['hem', 'GT'] columne

        hard_neg_df = hard_df[hard_df['GT'] == self.IB_CLASS]
        hard_pos_df = hard_df[hard_df['GT'] == self.OOB_CLASS]
        
        vanila_neg_df = vanila_df[vanila_df['GT'] == self.IB_CLASS]
        vanila_pos_df = vanila_df[vanila_df['GT'] == self.OOB_CLASS]

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df        
    
    def extract_hem_from_softmax_diff(self,
                                      dropout_predictions,
                                      gt_list,
                                      img_path_list,
                                      hem_type):
        hem_idx, vanila_idx = list(), list()
        cols = ['Img_path', 'GT']
        
        # 맞았는지, 틀렸는지 (answer) - voting 방식.
        answer, _ = self.get_answer(dropout_predictions, gt_list, answer_type=None)
        answer_list = answer.tolist()
        
        predict_diff = np.diff(dropout_predictions, axis=2) # shape (forward_passes, n_samples, 1) // (3, 550, 1) 
        predict_abs = np.abs(predict_diff) # shape (forward_passes, n_samples, 1) // (3, 550, 1) 
        predict_mean = np.mean(predict_abs, axis=0) # shape (n_samples, 1) // (550, 1) 
        predict_table = np.squeeze(predict_mean) # shape (n_samples, ) // (550, )

        vanila_df = pd.DataFrame([x for x in zip(img_path_list, gt_list)], columns=cols)
        top_k = int(len(vanila_df) * self.args.top_ratio)

        if hem_type == 'small': # lower case
            hard_idx = predict_table.argsort()[:top_k].tolist() # 올림차순
        elif hem_type == 'large': # upper case
            hard_df_upper_idx = predict_table.argsort()[-top_k:].tolist() # 내림차순

            hard_idx = []
            for i, answer in enumerate(answer_list): # answer_list
                if answer == False and (i in hard_df_upper_idx): # faster
                    hard_idx.append(i)
               
        all_ids = set(list(range(len(vanila_df))))
        hard_ids = set(hard_idx)
        
        non_hard_ids = list(all_ids.difference(hard_ids))
               
        hard_df = vanila_df.loc[hard_idx, :].reset_index(drop=True)
        vanila_df = vanila_df.loc[non_hard_ids, :].reset_index(drop=True)
        
        hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.split_to_hem_vanila_df(hard_df, vanila_df)

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df
        
    def cal_mutual_info(self, dropout_predictions):
        # dropout_predictinos (1, 46086, 2)
        # H(X), H(Y)
        epsilon = sys.float_info.min
        entropy1 = -np.sum(dropout_predictions * np.log2(dropout_predictions + epsilon), axis=0) # (46086, 2)
        entropy1_sum = np.sum(entropy1, axis=1) # (46086, )
        
        # H(X, Y)
        dropout_multi = dropout_predictions[:,:,0] * dropout_predictions[:,:,1] # (1, 46086)
        dropout_multi = np.transpose(dropout_multi) # (46086, 1)
        entropy2 = -np.sum(dropout_multi * np.log2(dropout_multi + epsilon), axis=1) 

        # H(X) + H(Y) - H(X, Y)
        mi_score = entropy1_sum - entropy2
        
        return mi_score
    
    def extract_hem_from_mutual_info(self, 
                                     dropout_predictions, 
                                     gt_list, 
                                     img_path_list, 
                                     hem_type):
        hem_idx = list()
        
        # answer = predict_np == np.array(gt_list) # compare with gt list
        answer, predict_np = self.get_answer(dropout_predictions, gt_list, answer_type='mean')
        predict_list = predict_np.tolist()
        
        mutual_info = self.cal_mutual_info(dropout_predictions)

        # sort mi index & extract top/btm sample index 
        top_k = int(len(mutual_info) * self.args.top_ratio)
        sorted_mi_index = mutual_info.argsort() # desecnding index
        
        if hem_type == 'small':
            btm_mi_index = sorted_mi_index[:top_k] # lowest
            
            wrong_idx = np.where(answer == False) # wrong example
            wrong_idx = wrong_idx[0].tolist() # remove return turple
            
            hem_idx += np.intersect1d(wrong_idx, btm_mi_index).tolist()
        elif hem_type == 'large':
            hem_idx = sorted_mi_index[-top_k:].tolist() # highest 
        
        # 2. split hem/vanila 
        total_df_dict = {
            'Img_path': img_path_list,
            'predict': predict_list,
            'GT': gt_list,
            'mi': mutual_info.tolist(),
            'hem': [self.NON_HEM] * len(img_path_list) # init hem
        }

        total_df = pd.DataFrame(total_df_dict)
        total_df.loc[hem_idx, ['hem']] = self.HEM # hem index
        hard_df = total_df[total_df['hem'] == self.HEM]
        vanila_df = total_df[total_df['hem'] == self.NON_HEM]

        hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.split_to_hem_vanila_df(hard_df, vanila_df)

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df
    
    def extract_hem_from_voting(self, 
                                dropout_predictions, 
                                gt_list, 
                                img_path_list, 
                                hem_type=None):
        hem_idx = list()
        
        # 1. extract hem index
        predict_table = np.argmax(dropout_predictions, axis=2) # (forward_passes, n_samples)
        predict_ratio = np.mean(predict_table, axis=0) # (n_samples)

        predict_np = np.around(predict_ratio).astype('int8') # threshold == 0.5, if predict_ratio >= 0.5, predict_class == OOB(1)
        predict_list = predict_np.tolist() # to list

        answer = predict_np == np.array(gt_list) # compare with gt list

        hem_idx = np.where(answer == False) # hard example
        hem_idx = hem_idx[0].tolist() # remove return turple

        # 2. split hem/vanila 
        total_df_dict = {
            'Img_path': img_path_list,
            'predict': predict_list,
            'GT': gt_list,
            'voting': predict_ratio.tolist(),
            'hem': [self.NON_HEM] * len(img_path_list) # init hem
        }

        total_df = pd.DataFrame(total_df_dict)
        total_df.loc[hem_idx, ['hem']] = self.HEM # hem index
        hard_df = total_df[total_df['hem'] == self.HEM]
        vanila_df = total_df[total_df['hem'] == self.NON_HEM]

        hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.split_to_hem_vanila_df(hard_df, vanila_df)

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df