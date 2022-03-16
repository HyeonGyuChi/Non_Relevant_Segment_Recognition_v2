import pandas as pd
import numpy as np
import natsort
import collections
import torch
from itertools import groupby, accumulate
from torch.utils.data import Sampler

pd.options.mode.chained_assignment = None


class BoundarySampler():
    def __init__(self, IB_ratio, WS_ratio):
        self.IB_ratio = IB_ratio
        self.WS_ratio = WS_ratio

    def sample(self, data):
        '''
            ##### 1. Calculate NRS (Non Related Surgery) start_idx, end_idx [[30, 38], [50, 100], [150, 157], ... ,[497679, 497828]] 
            nrs_start_end_idx_list = self.calc_nrs_idx()

            ##### 2. Select RS (Wise-Related Surgery) idx
            self.assets_df = self.extract_wise_rs_idx(nrs_start_end_idx_list)

            ##### 3. Set ratio
            self.final_assets = self.set_ratio()
        '''
        
        data['patient'] = data.img_path.str.split('/').str[4]
        patient_list = list(set(data['patient']))
        
        # 전체 환자에 대한 리스트
        self.class_idx_list = data['class_idx'].tolist()
        self.class_idx_len = len(self.class_idx_list)

        print('\n\n[TOTAL GT] class_idx_len : {}\n\n'.format(self.class_idx_len))

        final_patients_list = []
        for patient in patients_list:
            patient_df = data[data['patient']==patient]

            ##### 1. Calculate NRS (Non Related Surgery) start_idx, end_idx [[30, 38], [50, 100], [150, 157], ... ,[497679, 497828]] per patient
            nrs_start_end_idx_list = self.calc_nrs_idx(patient_df['class_idx'])

            ##### 2. Select RS (Wise-Related Surgery) idx per patient
            patient_df = self.extract_wise_rs_idx(patient_df, nrs_start_end_idx_list)

            final_patients_list.append(patient_df)

        final_patients_df = pd.concat(final_patients_list)

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
        runlength_df = runlength_df.append(encode_df)

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

        # print('==================== FINAL self.assets_df ====================')
        # print(patient_df, '\n\n')

        # # pd.set_option('display.max_row', None)
        # print('==================== Wise RS ====================')
        # print(patient_df[patient_df['wise_rs'] == True], '\n\n')

        return patient_df

    def set_ratio(self, final_patients_df):
        ##### 3. Set ratio
        assets_nrs_df = final_patients_df[final_patients_df['class_idx']==1]

        assets_wise_rs_df = final_patients_df[final_patients_df['wise_rs']==True]
        
        try:
            assets_vanila_df = final_patients_df[(final_patients_df['wise_rs']==False) & (final_patients_df['class_idx']==0)].sample(n=int(len(assets_nrs_df)*self.IB_ratio-len(assets_wise_rs_df)), replace=False, random_state=self.random_seed)
        except:
            assets_vanila_df = final_patients_df[(final_patients_df['wise_rs']==False) & (final_patients_df['class_idx']==0)].sample(frac=1, replace=False, random_state=self.random_seed)

        assets_rs_df = pd.concat([assets_wise_rs_df, assets_vanila_df]).sample(frac=1, replace=False, random_state=self.random_seed).reset_index(drop=True)

        final_assets = pd.concat([assets_nrs_df, assets_rs_df]).sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        print('\nself.assets_nrs_df\n', assets_nrs_df)
        print('\nself.assets_rs_df\n', assets_rs_df)

        print('\nself.final_assets\n', final_assets)

        return final_assets
        
        
class OverSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = self.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            np.random.shuffle(self.labels)

            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.m_per_class] = self.safe_random_choice(
                    t, size=self.m_per_class
                )
                i += self.m_per_class
        return iter(idx_list)

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1

    def safe_random_choice(self, input_data, size):
        """
        Randomly samples without replacement from a sequence. It is "safe" because
        if len(input_data) < size, it will randomly sample WITH replacement
        Args:
            input_data is a sequence, like a torch tensor, numpy array,
                            python list, tuple etc
            size is the number of elements to randomly sample from input_data
        Returns:
            An array of size "size", randomly sampled from input_data
        """
        replace = len(input_data) < size
        return np.random.choice(input_data, size=size, replace=replace)

    def get_labels_to_indices(self, labels):
        """
        Creates labels_to_indices, which is a dictionary mapping each label
        to a numpy array of indices that will be used to index into self.dataset
        """
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        labels_to_indices = collections.defaultdict(list)
        for i, label in enumerate(labels):
            labels_to_indices[label].append(i)
        for k, v in labels_to_indices.items():
            labels_to_indices[k] = np.array(v, dtype=np.int)
        return labels_to_indices