import pandas as pd
import numpy as np
import natsort
import collections
import torch
from itertools import groupby, accumulate

pd.options.mode.chained_assignment = None


class BoundarySampler():
    def __init__(self, IB_ratio, WS_ratio):
        self.IB_ratio = IB_ratio
        self.WS_ratio = WS_ratio

    # def sample(self, data, patients_list):
    def sample(self, data):
        '''
            ##### 1. Calculate NRS (Non Related Surgery) start_idx, end_idx [[30, 38], [50, 100], [150, 157], ... ,[497679, 497828]] 
            nrs_start_end_idx_list = self.calc_nrs_idx()

            ##### 2. Select RS (Wise-Related Surgery) idx
            self.assets_df = self.extract_wise_rs_idx(nrs_start_end_idx_list)

            ##### 3. Set ratio
            self.final_assets = self.set_ratio()
        '''
        
        
        # 전체 환자에 대한 리스트
        self.class_idx_list = data['class_idx'].tolist()
        self.class_idx_len = len(self.class_idx_list)
        # print("self.class_idx_list",self.class_idx_list)
        # print("self.class_idx_len",self.class_idx_len)

        # print('[TOTAL GT] class_idx_len : {}'.format(self.class_idx_len))
        
        patient_df = data
        # print("first patient_df\n",patient_df)
        ##### 1. Calculate NRS (Non Related Surgery) start_idx, end_idx [[30, 38], [50, 100], [150, 157], ... ,[497679, 497828]] per patient
        # print("patient_df['class_idx']\n",patient_df['class_idx'])
        nrs_start_end_idx_list = self.calc_nrs_idx(patient_df['class_idx'])
        
        ##### 2. Select RS (Wise-Related Surgery) idx per patient
        patient_df = self.extract_wise_rs_idx(patient_df, nrs_start_end_idx_list)
        # print("second patient_df\n",patient_df)

        final_patients_df = patient_df[['img_path', 'class_idx', 'wise_rs']] 
        # print("final_patients_df\n",final_patients_df)

        return self.set_ratio(final_patients_df)
        
    def encode_list(self, s_list): # run-length encoding from list
        return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]

    def wise_rs_parity_check(self, wise_rs_idx, patient_df):
        return len(wise_rs_idx) == len(patient_df)

    def calc_nrs_idx(self, class_idx_list_per_patient):
        ##### 1. Calculate NRS (Non Related Surgery) start_idx, end_idx [[30, 38], [50, 100], [150, 157], ... ,[497679, 497828]] 
        encode_data = self.encode_list(class_idx_list_per_patient)
        encode_df = pd.DataFrame(data=encode_data, columns=['length', 'class']) # [length, value]
        
        # arrange data
        runlength_df = pd.DataFrame(range(0,0)) # empty df
        runlength_df = runlength_df.append(encode_df) #pd.concat로 수정 권장 (warning)


        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_gt = runlength_df['length'] # run length info of gt

        # data processing for barchart
        data = np.array(runlength_gt.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        runlength_df['accum'] = data_cum # accumulation of gt
        
        nrs_runlength_df = runlength_df[runlength_df['class'] == 1]

        nrs_runlength_df['start_idx'] = nrs_runlength_df['accum'] - nrs_runlength_df['length']
        nrs_runlength_df['end_idx'] = nrs_runlength_df['accum'] - 1

        start_idx_list = nrs_runlength_df['start_idx'].tolist()
        end_idx_list = nrs_runlength_df['end_idx'].tolist()

        nrs_start_end_idx_list = []
        for start_idx, end_idx in zip(start_idx_list, end_idx_list):
            nrs_start_end_idx_list.append([start_idx, end_idx])

        # print('\n\n================== NRS_START_END_IDX_LIST (len:{}) ================== \n\n{}\n\n'.format(len(nrs_start_end_idx_list), nrs_start_end_idx_list))

        return nrs_start_end_idx_list

    def extract_wise_rs_idx(self, patient_df, nrs_start_end_idx_list):
        ##### 2. Extract RS (Wise-Related Surgery) idx
        wise_rs_idx = [False] * len(patient_df)

        for nrs_idx in nrs_start_end_idx_list:
            nrs_start_idx = nrs_idx[0]
            nrs_end_idx = nrs_idx[1]

            start_end_gap = nrs_end_idx-nrs_start_idx
            wise_window_size = int((start_end_gap//self.WS_ratio) * self.IB_ratio) # start_end_gap <= 4 (default) -> wise_window_size = 0 

            if nrs_start_idx == 0: # nrs start idx == 0 이면, 그 이전의 프레임을 선택할 수 없음. 
                pass
            elif nrs_start_idx-wise_window_size < 0: # nrs start idx != 0 인데, gap 을 뺀 후가 0보다 작다면, 0 ~ nrs_start_idx select. 
                wise_rs_idx[0:nrs_start_idx] = [True]*len(wise_rs_idx[0:nrs_start_idx])
            else: 
                wise_rs_idx[nrs_start_idx-wise_window_size:nrs_start_idx] = [True]*len(wise_rs_idx[nrs_start_idx-wise_window_size:nrs_start_idx])
            
            if nrs_end_idx+1 == len(patient_df): # nrs end idx + 1 == len(patient_df)  이면, len(patient_df) 을 넘어선 프레임을 선택할 수 없음. 
                pass
            elif nrs_end_idx+wise_window_size+1 > len(patient_df): # nrs end idx + 1 != len(patient_df) 인데, gap 을 추가한 후가 len(patient_df) 보다 크다면, nrs_end_idx+1 ~ 끝까지 select.
                wise_rs_idx[nrs_end_idx+1:len(patient_df)] = [True]*len(wise_rs_idx[nrs_end_idx+1:len(patient_df)])
            else:
                wise_rs_idx[nrs_end_idx+1:nrs_end_idx+wise_window_size+1] = [True]*len(wise_rs_idx[nrs_end_idx+1:nrs_end_idx+wise_window_size+1])

        ## Parity check of wise_rs_idx, len(patient_df).
        if not self.wise_rs_parity_check(wise_rs_idx, patient_df):
            raise Exception("\nERROR: NOT MATCH BETWEEN len(wise_rs_idx), class_idx_len ====> len(wise_rs_idx): {}, len(patient_df): {}".format(len(wise_rs_idx), len(patient_df)))
        
        
        ##### 2-1. Consensus between wise_rs_idx(rs+nrs) and patient_df['class_idx'] (rs)
        final_wise_rs_idx = []
        for wise_rs, gt in zip(wise_rs_idx, patient_df['class_idx']):
            if (wise_rs==True) and (gt==0):
                final_wise_rs_idx.append(True)
            else:
                final_wise_rs_idx.append(False)

        patient_df['wise_rs'] = final_wise_rs_idx

        return patient_df

    def set_ratio(self, final_patients_df):
        # print("start set_ratio")
        ##### 3. Set ratio
        assets_nrs_df = final_patients_df[final_patients_df['class_idx']==1]

        assets_wise_rs_df = final_patients_df[final_patients_df['wise_rs']==True]
        
        try:
            assets_vanila_df = final_patients_df[(final_patients_df['wise_rs']==False) & (final_patients_df['class_idx']==0)].sample(n=int(len(assets_nrs_df)*self.IB_ratio-len(assets_wise_rs_df)), replace=False)
        except:
            assets_vanila_df = final_patients_df[(final_patients_df['wise_rs']==False) & (final_patients_df['class_idx']==0)].sample(frac=1, replace=False)

        assets_rs_df = pd.concat([assets_wise_rs_df, assets_vanila_df]).sample(frac=1, replace=False).reset_index(drop=True)

        final_assets = pd.concat([assets_nrs_df, assets_rs_df]).sample(frac=1).reset_index(drop=True)

        # print('\nself.assets_nrs_df\n', assets_nrs_df)
        # print('\nself.assets_rs_df\n', assets_rs_df)

        return final_assets