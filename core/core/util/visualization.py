import os
import math
import pandas as pd
import numpy as np
import natsort
import datetime
from itertools import groupby

import matplotlib.pyplot as plt
from core.util.metric import MetricHelper



class VisualTool():
    # TODO visual sampling, multi 추가 필요
    def __init__(self, args):
        self.args = args
        self.result_path = None
        self.metric_helper = MetricHelper(self.args)
        
        self.window_size = 300
        self.section_num = 2
        
        self.EXCEPTION_NUM = -100
        self.RS_CLASS, self.NRS_CLASS = (0,1)
        self.SSIM_CLASS=2
        self.FN_CLASS, self.FP_CLASS = (3,4) # only use in visual predict
        
        
        # only use in calc section sampling
        self.NEG_HARD_CLASS, self.POS_HARD_CLASS, self.NEG_VANILA_CLASS, self.POS_VANILA_CLASS = (2,3,4,5) 
        
    def set_result_path(self, result_path):
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
            
            if not os.path.isdir(dpath):
                continue
            
            if patient not in path_dict:
                path_dict[patient] = []
            
            for video_name in natsort.natsorted(os.listdir(dpath)):
                if 'png' in video_name:
                    continue
                    
                csv_path = os.path.join(dpath, video_name, f'{video_name}.csv')            
                path_dict[patient].append(csv_path)
        
        for patient, path_list in path_dict.items():
            gt_list = []
            predict_list = []
            
            for csv_path in path_list:
                # data = pd.read_csv(csv_path).values[2:4]
                data = pd.read_csv(csv_path)
                # data = data[['frame_idx', 'predict', 'gt', 'target_img']]
                data = data[['predict', 'gt']].values
                data = np.array(data).astype('uint8')
                
                gt_list += list(data[:, 1])
                predict_list += list(data[:, 0])
                
            save_path = '{}/{}/{}.png'.format(self.result_path, patient, patient)            
            self.visualize_inference(gt_list, predict_list, save_path)
        
            
    def visualize_inference(self, gt_list, predict_list, save_path):
        fig, ax = self._get_plt('predict')
        
        metrics = self.calc_metrics_with_index(gt_list, predict_list)
        metrics_per_section = self.calc_section_metrics(gt_list, predict_list)
        
        frame_label = list(range(0, len(gt_list) * self.args.inference_interval, self.args.inference_interval))
        time_label = [self.idx_to_time(idx, fps=30) for idx in frame_label]
        
        label_names = ['RS', 'NRS','SSIM','FN', 'FP']
        colors = ['cadetblue', 'orange', 'white', 'blue', 'red']
        yticks = ['GT', 'PREDICT'] # y축 names, 순서중요
        height = 0.5 # bar chart thic
        
        visual_data = {
            'GT': gt_list,
            'PREDICT': predict_list,
        }
        
        # for visualize FP, FN // change class
        predict_df = pd.DataFrame(predict_list)
        predict_df.iloc[metrics['FN_idx'],] = self.FN_CLASS
        predict_df.iloc[metrics['FP_idx'],] = self.FP_CLASS
        visual_data['PREDICT'] = list(np.array(predict_df[0].tolist()))

        encode_data = {}
        runlength_df = pd.DataFrame(range(0,0)) # empty df
        
        # arrange data
        for y_name in yticks: # run_length coding
            encode_data[y_name] = pd.DataFrame(data=self.encode_list(visual_data[y_name]), 
                                               columns=[y_name, 'class']) # [length, value]
            runlength_df = runlength_df.append(encode_data[y_name])

        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_model = runlength_df[yticks] # run length info of model prediction
        
        # data processing for barchart
        data = np.array(runlength_model.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        ### draw ###
        ##### initalize label for legned, this code should be write before writing barchart #####
        for _cls in [self.RS_CLASS, self.NRS_CLASS, self.SSIM_CLASS, self.FN_CLASS, self.FP_CLASS]:
            # print("cls",_cls)
            init_bar = ax[0].barh(range(len(yticks)), 
                            np.zeros(len(yticks)), 
                              label=label_names[_cls], 
                              height=height, 
                              color=colors[_cls]) # dummy data
        
        # draw bar
        for i, frame_class in enumerate(runlength_class):
            widths = data[i,:]
            starts = data_cum[i,:] - widths
            
            bar = ax[0].barh(range(len(yticks)), widths, 
                             left=starts, height=height, 
                             color=colors[frame_class]) # don't input label

        # write text on PREDICT bar
        gt_oob_ratio = metrics['gt_OOB'] / (metrics['gt_IB'] + metrics['gt_OOB'])
        predict_oob_ratio = metrics['predict_OOB'] / (metrics['predict_IB'] + metrics['predict_OOB'])
        text_bar = ax[0].barh(1, 0, height=height) # dummy data        
        
        self.present_text(ax[0], text_bar, 
                          'CR : {:.3f} | OR : {:.3f} | JACCARD: {:.3f} | OOB_RATIO(GT) : {:.2f} | OOB_RATIO(PD) : {:.2f}'.format(metrics['CR'], metrics['OR'], 
                                                                                                                                metrics['Jaccard'], 
                                                                                                                                gt_oob_ratio, predict_oob_ratio))

        ### write on figure 
        # set title
        patient_name = save_path.split('/')[-1][:-4]
        model_name = save_path.split('/')[-4].split('-')[0]
        title_name = 'Predict of {}'.format(patient_name)
        sub_title_name = 'model: {} | inferene interval: {} | windows size: {} | section num: {}'.format(model_name, 
                                                                                                         self.args.inference_interval, 
                                                                                                         self.window_size, 
                                                                                                         self.section_num)
        fig.suptitle(title_name, fontsize=16)
        ax[0].set_title(sub_title_name)

        # set xticks pre section size
        ax[0].set_xticks(range(0, len(frame_label), self.window_size))
        
        xtick_labels = ['{}\n{}'.format(time, frame) if i_th % 2 == 0 else '\n\n{}\n{}'.format(time, frame) \
                        for i_th, (time, frame) in enumerate(zip(frame_label[::self.window_size], time_label[::self.window_size]))]
        
        ax[0].set_xticklabels(xtick_labels) # xtick change
        ax[0].xaxis.set_tick_params(labelsize=6)
        ax[0].set_xlabel('Frame / Time (h:m:s:fps)', fontsize=12)

        # set yticks
        ax[0].set_yticks(range(len(yticks)))
        ax[0].set_yticklabels(yticks, fontsize=10)	
        ax[0].set_ylabel('Model', fontsize=12)

        # 8. legend
        box = ax[0].get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax[0].legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        # 9. 보조선(눈금선) 나타내기
        ax[0].set_axisbelow(True)
        ax[0].xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

        # 10. draw subplot ax (section metrics)
        section_start_idx = (np.array(metrics_per_section['start_idx']) * self.args.inference_interval).tolist()
        CR_value, OR_value = metrics_per_section['section_CR'], metrics_per_section['section_OR']

        CR_value = [1.0 if val==self.EXCEPTION_NUM else val for val in CR_value] # -100 EXP 일 경우 1로 처리
        OR_value = [0.0 if val==self.EXCEPTION_NUM else val for val in OR_value] # -100 EXP 일 경우 0로 처리
        
        self.draw_plot(ax[1], 'CR of Predict', section_start_idx, CR_value, y_min=-1.0, y_max=1.0, color='blue')
        self.draw_plot(ax[2], 'OR of Predict', section_start_idx, OR_value, y_min=0.0, y_max=1.0, color='red')

        ### file write        
        plt.show()
        plt.savefig(save_path, format='png', dpi=500)
        plt.close(fig)
        
    def _get_plt(self, visual_type):
        assert visual_type in ["compare",'predict', 'predict_multi', 'sampling'], 'NOT SUPPORT VISUAL MODE'

        if visual_type == 'compare':
            fig, ax = plt.subplots(1,1,figsize=(18,15)) # 1x1 figure matrix 생성, figsize=(가로, 세로) 크기지정

            plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0, 
                    hspace=0.35)
                    
        if visual_type == 'predict':
            fig, ax = plt.subplots(3,1,figsize=(18,15)) # 1x1 figure matrix 생성, figsize=(가로, 세로) 크기지정

            plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0, 
                    hspace=0.35)

        if visual_type == 'predict_multi':
            fig, ax = plt.subplots(1,1,figsize=(30,7)) # 1x1 figure matrix 생성, figsize=(가로, 세로) 크기지정

            plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0, 
                    hspace=0.35)

        elif visual_type == 'sampling':
            fig, ax = plt.subplots(2,1,figsize=(18,14)) # 2x1 figure matrix 생성, 가로(18인치)x세로(13인치) 크기지정

        return fig, ax
        
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
        slide_window_start_end_idx= [[start_idx * self.window_size, (start_idx + self.section_num) * self.window_size] \
                                        for start_idx in range(math.ceil(total_len/self.window_size))] # overlapping section

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
 
    # for section metric
    def draw_plot(self, ax, title, x_value, y_value, y_min, y_max, color='blue'):
        ax.plot(x_value, y_value, marker='o', markersize=4, alpha=1.0, color=color)

        # set title
        ax.set_title(title)
        
        # set x ticks
        ax.set_xticks(x_value)
        xtick_labels = ['{}'.format(frame) if i_th % 2 == 0 else '\n{}'.format(frame) for i_th, frame in enumerate(x_value)]
        ax.set_xticklabels(xtick_labels) # xtick change
        ax.xaxis.set_tick_params(labelsize=6)
        ax.set_xlabel('Start Frame', fontsize=12)
        
        # set y ticks
        ax.set_ylim(ymin=y_min, ymax=y_max)

        # 보조선
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)
 
    def idx_to_time(self, idx, fps):
        time_s = idx // fps
        frame = int(idx % fps)

        converted_time = str(datetime.timedelta(seconds=time_s))
        converted_time = converted_time + '.' + str(frame / fps)

        return converted_time
    
    def encode_list(self, s_list): # run-length encoding from list
        return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]


    def visualize_compare(self, patient_labels,save_path):
        fig, ax = self._get_plt('compare')
        
        patient = list(patient_labels.keys())[0]
        labels = list(patient_labels.values())[0]
        gt_list = list(labels.values())[0]
        predict_list = list(labels.values())[1]
        ssim_list = list(labels.values())[2]
        
        metrics = self.calc_metrics_with_index(gt_list, predict_list)
        metrics_per_section = self.calc_section_metrics(gt_list, predict_list)
        
        frame_label = list(range(0, len(gt_list) * self.args.inference_interval, self.args.inference_interval))
        time_label = [self.idx_to_time(idx, fps=30) for idx in frame_label]
        
        label_names = ['RS', 'NRS', "SSIM",'FN', 'FP',]
        colors = ['cadetblue', 'orange', "white",'blue', 'red']
        yticks = ['GT', 'PREDICT',"SSIM"] # y축 names, 순서중요
        height = 0.5 # bar chart thic
        
        visual_data = {
            'GT': gt_list,
            'PREDICT': predict_list,
        }
        
        # for visualize FP, FN // change class
        predict_df = pd.DataFrame(predict_list)
        ssim_df = pd.DataFrame(ssim_list)
        predict_df.iloc[metrics['FN_idx'],] = self.FN_CLASS
        predict_df.iloc[metrics['FP_idx'],] = self.FP_CLASS
        visual_data['PREDICT'] = list(np.array(predict_df[0].tolist()))
        visual_data['SSIM'] = list(np.array(ssim_df[0].tolist()))



        encode_data = {}
        runlength_df = pd.DataFrame(range(0,0)) # empty df
        
        # arrange data
        for y_name in yticks: # run_length coding
            encode_data[y_name] = pd.DataFrame(data=self.encode_list(visual_data[y_name]), 
                                            columns=[y_name, 'class']) # [length, value]
            runlength_df = runlength_df.append(encode_data[y_name])

        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_model = runlength_df[yticks] # run length info of model prediction
        
        # data processing for barchart
        data = np.array(runlength_model.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        ### draw ###
        ##### initalize label for legned, this code should be write before writing barchart #####
        for _cls in [self.RS_CLASS, self.NRS_CLASS, self.SSIM_CLASS, self.FN_CLASS, self.FP_CLASS]:
            init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), 
                              label=label_names[_cls], height=height, 
                              color=colors[_cls]) # dummy data
        
        # draw bar
        for i, frame_class in enumerate(runlength_class):
            widths = data[i,:]
            starts = data_cum[i,:] - widths
            
            bar = ax.barh(range(len(yticks)), widths, 
                             left=starts, height=height, 
                             color=colors[frame_class]) # don't input label

        # write text on PREDICT bar
        gt_oob_ratio = metrics['gt_OOB'] / (metrics['gt_IB'] + metrics['gt_OOB'])
        predict_oob_ratio = metrics['predict_OOB'] / (metrics['predict_IB'] + metrics['predict_OOB'])
        text_bar = ax.barh(1, 0, height=height) # dummy data        
        
        self.present_text(ax, text_bar, 
                          'CR : {:.3f} | OR : {:.3f} | JACCARD: {:.3f} | OOB_RATIO(GT) : {:.2f} | OOB_RATIO(PD) : {:.2f}'.format(metrics['CR'], metrics['OR'], 
                                                                                                                                metrics['Jaccard'], 
                                                                                                                                gt_oob_ratio, predict_oob_ratio))

        ### write on figure 
        # set title
        patient_name = patient
        model_name = self.args.model
        title_name = 'Predict of {}'.format(patient_name)
        sub_title_name = 'model: {} | inferene interval: {} | windows size: {} | section num: {}'.format(model_name, 
                                                                                                         self.args.inference_interval, 
                                                                                                         self.window_size, 
                                                                                                         self.section_num)
        fig.suptitle(title_name, fontsize=16)
        ax.set_title(sub_title_name)


        # set yticks
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=10)	
        ax.set_ylabel('Model', fontsize=12)

        # 8. legend
        box = ax.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        # 9. 보조선(눈금선) 나타내기
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)


        ### file write        
        plt.show()
        plt.savefig(save_path, format='png', dpi=500)
        plt.close(fig)
        