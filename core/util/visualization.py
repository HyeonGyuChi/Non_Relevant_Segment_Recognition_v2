import os
import math
import pandas as pd
import numpy as np
import natsort
import datetime

import matplotlib.pyplot as plt
from core.util.metric import MetricHelper



class VisualTool():
    def __init__(self, args):
        self.args = args
        self.result_path = None
        self.metric_helper = MetricHelper(self.args)
        
        self.window_size = 300
        self.section_num = 2
        
        self.RS_CLASS, self.NRS_CLASS = (0,1)
        # only use in calc section sampling
        self.NEG_HARD_CLASS, self.POS_HARD_CLASS, self.NEG_VANILA_CLASS, self.POS_VANILA_CLASS, = (2,3,4,5) 
        
    def set_path(self, result_path):
        # ex) : logs/*~~~~~~*/inference_results
        self.result_path = result_path
        
    def set_window_size(self, window_size):
        self.window_size = window_size
        
    def set_section_num(self, section_num):
        self.section_num = section_num
        
    # base visualization
    def visualize(self):
        if self.result_path is None:
            raise 'Not found csv path'
        
        path_dict = {}
        
        for patient in natsort.natsorted(os.listdir(self.result_path)):
            dpath = os.path.join(self.result_path, patient)
            
            if patient not in path_dict:
                path_dict[patient] = []
            
            for video_name in natsort.natsorted(os.listdir(dpath)):
                csv_path = os.path.join(dpath, video_name, f'{video_name}.csv')            
                path_dict[patient].append(csv_path)
        
        for patient, path_list in path_dict.items():
            gt_list = []
            predict_list = []
            
            for csv_path in path_list:
                data = pd.read_csv(csv_path).values[2:4]
                data = np.array(data).astype('uint8')
                
                gt_list += list(data[1])
                predict_list += list(data[0])
                
            
        
        
            
    def visualize_inference(self, gt_list, predict_list):
        label_names = ['RS', 'NRS', 'FN', 'FP']
        colors = ['cadetblue', 'orange', 'blue', 'red']
        height = 0.5 # bar chart thic
        
        metrics = self.calc_metrics_with_index(gt_list, predict_list)
        metrics_per_section = self.calc_section_metrics(gt_list, predict_list)
        
        frame_label = list(range(0, len(gt_list) * self.args.inference_interval, self.args.inference_interval))
        time_label = [self.visual_helper.idx_to_time(idx, fps=30) for idx in frame_label]
        yticks = ['GT', 'PREDICT'] # y축 names, 순서중요

        visual_data = {
            'GT':gt_list,
            'PREDICT':predict_list,
        }
        
        
    
            
    
    
    
    
    
    
    
    
    def calc_section_metrics(self, gt_list, predict_list):
        metrics_per_section = {
            'start_idx':[],
            'end_idx':[],
            'section_CR':[],
            'section_OR':[],
        }

        data = {
            'GT': gt_list,
            'PREDICT': predict_list,
        }

        total_info_df = pd.DataFrame(data)
        total_len = len(total_info_df)
        slide_window_start_end_idx= [[start_idx * self.window_size, (start_idx + self.section_num) * self.window_size] for start_idx in range(math.ceil(total_len/self.window_size))] # overlapping section

        # calc metric per section
        for start, end in slide_window_start_end_idx : # slicing            
            section_df = total_info_df.iloc[start:end, ]

            section_gt_list, section_predict_list = section_df['GT'].tolist(), section_df['PREDICT'].tolist()
            section_metrics = self.calc_metrics_with_index(section_gt_list, section_predict_list)
            
            end = start + len(section_df) - 1 # truly end idx

            metrics_per_section['start_idx'].append(start)
            metrics_per_section['end_idx'].append(end)
            metrics_per_section['section_CR'].append(section_metrics['CR'])
            metrics_per_section['section_OR'].append(section_metrics['OR'])

        return metrics_per_section
    
    
    def calc_metrics_with_index(self, gt_list, predict_list):
        self.metric_helper.write_preds(np.array(predict_list), np.array(gt_list))
        metrics = self.metric_helper.calc_metric()
        
        # with metric index
        gt_np = np.array(gt_list)
        predict_np = np.array(predict_list)

        metrics['TP_idx'] = np.where((predict_np == self.NRS_CLASS) & (gt_np == self.NRS_CLASS))[0].tolist()
        metrics['TN_idx'] = np.where((predict_np == self.RS_CLASS) & (gt_np == self.RS_CLASS))[0].tolist()
        metrics['FP_idx'] = np.where((predict_np == self.NRS_CLASS) & (gt_np == self.RS_CLASS))[0].tolist()
        metrics['FN_idx'] = np.where((predict_np == self.RS_CLASS) & (gt_np == self.NRS_CLASS))[0].tolist()

        return metrics
        
    # for text on bar
    def present_text(self, ax, bar, text, color='black'):
        for rect in bar:
            posx = rect.get_x()
            posy = rect.get_y() - rect.get_height()*0.1
            ax.text(posx, posy, text, color=color, rotation=0, ha='left', va='bottom')
 
    def idx_to_time(self, idx, fps):
        time_s = idx // fps
        frame = int(idx % fps)

        converted_time = str(datetime.timedelta(seconds=time_s))
        converted_time = converted_time + '.' + str(frame / fps)

        return converted_time