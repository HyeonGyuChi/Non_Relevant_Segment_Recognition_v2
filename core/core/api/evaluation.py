import os
import json
import numpy as np
import pandas as pd

from core.util.metric import MetricHelper
from core.util.report import Reporter


class Evaluator():
    def __init__(self, args):
        self.args = args
        self.base_save_path = self.args.save_path + '/inference_results'
        self.inference_interval = 1 #self.args.inference_interval
        self.metric_helper = MetricHelper(self.args)
        self.reporter = Reporter()
        
    def set_inference_interval(self, inference_interval):
        self.inference_interval = inference_interval
    
    def set_path(self, output_path):
        self.output_path = output_path
        self.get_assets()
        
    def set_assets(self, predict_df):
        self.predict_list = list(predict_df['predict'].tolist())
        self.gt_list = list(predict_df['gt'].tolist())

    def get_assets(self): 
        predict_df = pd.read_csv(self.output_path)
        self.predict_list = list(predict_df['predict'].tolist())
        self.gt_list = list(predict_df['gt'].tolist())
        
    def calc(self):
        """
        Calculate Over_Estimation_Ratio(OR) and Confidence_Ratio(CR).
        {
            "01_G_01_R_100_ch1_03": {
                "over_estimation_ratio": 0.01046610685164902,
                "confidence_ratio": 0.9785809906291834
        }

        Returns:
            `turple`, of Over_Estimation_Ratio and Confidence_Ratio.
            [self.over_estimation_ratio, self.confidence_ratio]

        Example:
            >>> calc()
                metrics = {
                    'TP': 
                    'TN': 
                    'FP': 
                    'FN': 
                    'Accuracy': 
                    'Precision': 
                    'Recall': 
                    'F1-Score': 

                    'OOB_metric':
                    'Over_estimation':
                    'Under_estimation':
                    'Correspondence_estimation':
                    'UNCorrespondence_estimation':
                }
        """
        # 1. prepare data(assets)
        # 2. calc metric
        
        self.metric_helper.write_preds(np.array(self.predict_list), np.array(self.gt_list))
        metrics = self.metric_helper.calc_metric()

        return metrics
    
    def update_report(self, patient_no, video_name, target_metric, target='video'):
        if target == 'video':
            video_metrics = target_metric
            
            video_CR, video_OR = video_metrics['CR'], video_metrics['OR']
            video_TP, video_FP, video_TN, video_FN = video_metrics['TP'], video_metrics['FP'], video_metrics['TN'], video_metrics['FN']
            video_TOTAL = video_FP + video_TP + video_FN + video_TN

            video_gt_IB, video_gt_OOB, video_predict_IB, video_predict_OOB = video_metrics['gt_IB'], video_metrics['gt_OOB'], \
                                                                            video_metrics['predict_IB'], video_metrics['predict_OOB']
            video_jaccard = video_metrics['Jaccard']
            video_precision, video_recall = video_metrics['Precision'], video_metrics['Recall']

            print('\t => video_name: {}'.format(video_name))
            print('\t    video_CR: {:.3f} | video_OR: {:.3f}'.format(video_CR, video_OR))
            print('\t    video_TP: {} | video_FP: {} | video_TN: {} | video_FN: {}'.format(video_TP, video_FP, video_TN, video_FN))
            
            
            video_results_dict = self.reporter.add_videos_report(patient_no=patient_no, video_no=video_name, 
                                                                FP=video_FP, TP=video_TP, FN=video_FN, TN=video_TN, 
                                                                TOTAL=video_TOTAL, CR=video_CR, OR=video_OR, 
                                                                gt_IB=video_gt_IB, gt_OOB=video_gt_OOB, 
                                                                predict_IB=video_predict_IB, predict_OOB=video_predict_OOB, 
                                                                precision=video_precision, recall=video_recall, jaccard=video_jaccard)
            
            save_path = self.base_save_path + '/videos_report.csv'
            self.save_dict_to_csv(video_results_dict, save_path)
            
        elif target == 'patient':
            patient_metrics = target_metric
            
            patient_CR, patient_OR = patient_metrics['CR'], patient_metrics['OR']
            patient_TP, patient_FP, patient_TN, patient_FN = patient_metrics['TP'], patient_metrics['FP'], patient_metrics['TN'], patient_metrics['FN']
            patient_TOTAL = patient_FP + patient_TP + patient_FN + patient_TN

            patient_gt_IB, patient_gt_OOB, patient_predict_IB, patient_predict_OOB = patient_metrics['gt_IB'], patient_metrics['gt_OOB'], patient_metrics['predict_IB'], patient_metrics['predict_OOB']

            patient_jaccard = patient_metrics['Jaccard']

            patient_precision, patient_recall = patient_metrics['Precision'], patient_metrics['Recall']

            print('\t\t => patient_no: {}'.format(patient_no))
            print('\t\t    patient_CR: {:.3f} | patient_OR: {:.3f}'.format(patient_CR, patient_OR))
            print('\t\t    patient_TP: {} | patient_FP: {} | patient_TN: {} | patient_FN: {}'.format(patient_TP, patient_FP, patient_TN, patient_FN))

            # save patient metrics        
            patient_results_dict = self.reporter.add_patients_report(patient_no=patient_no, 
                                                                     FP=patient_FP, TP=patient_TP, FN=patient_FN, TN=patient_TN, 
                                                                     TOTAL=patient_TOTAL, CR=patient_CR, OR=patient_OR, 
                                                                     gt_IB=patient_gt_IB, gt_OOB=patient_gt_OOB, 
                                                                     predict_IB=patient_predict_IB, predict_OOB=patient_predict_OOB, 
                                                                     precision=patient_precision, recall=patient_recall, jaccard=patient_jaccard)
            
            save_path = self.base_save_path + '/patients_report.csv'
            self.save_dict_to_csv(patient_results_dict, save_path)
            
        elif target == 'all':
            patients_metrics_list = target_metric
            
             # for calc total patients CR, OR + (mCR, mOR)
            total_metrics = self.metric_helper.aggregate_calc_metric(patients_metrics_list)
            total_mCR, total_mOR, total_CR, total_OR = total_metrics['mCR'], total_metrics['mOR'], total_metrics['CR'], total_metrics['OR']
            total_mPrecision, total_mRecall = total_metrics['mPrecision'], total_metrics['mRecall']
            total_Jaccard = total_metrics['Jaccard']
                    
            self.reporter.set_experiment(model=self.args.model, 
                                methods=self.args.hem_extract_mode,
                                inference_fold=self.args.fold, 
                                mCR=total_mCR, mOR=total_mOR, 
                                CR=total_CR, OR=total_OR, 
                                mPrecision=total_mPrecision, mRecall=total_mRecall, 
                                Jaccard=total_Jaccard, 
                                details_path=self.base_save_path, 
                                model_path='') # model path는 우선 생략
                        
            self.reporter.save_report() # save report
    
    def evaluation(self, results_dict=None, output_path_list=None):
        
        # exception
        if results_dict is None and output_path_list is None:
            raise 'No input data!'
        
        patients_metrics_list = []
                
        report_path = self.base_save_path + '/Report.json'
        self.reporter.set_save_path(report_path)
        
        if results_dict is not None:
            for patient_no in results_dict.keys():
                videos_metrics_list = []
                patient_res = results_dict[patient_no]
                
                for video_name in patient_res.keys():
                    self.set_assets(patient_res[video_name])
                    
                    with open(self.base_save_path+"/terminal_logs_EVALUATION.txt","a") as f:
                        f.write("patient_no",patient_no)
                        f.write("video_name",video_name)
                    
                    # for calc patients metric
                    video_metrics = self.calc()
                    self.update_report(patient_no, video_name, video_metrics, target='video')
                    
                    videos_metrics_list.append(video_metrics)
                
                # calc each patients CR, OR
                patient_metrics = self.metric_helper.aggregate_calc_metric(videos_metrics_list)
                self.update_report(patient_no, video_name, patient_metrics, target='patient')
                
                # for calc total patients CR, OR
                patients_metrics_list.append(patient_metrics)
                
        elif output_path_list is not None: 
            patient_path_list = self.sort_for_patient(output_path_list)
            
            for patient_no, path_list in patient_path_list.items():
                videos_metrics_list = []
                
                for video_name, output_path in path_list:
                    self.set_path(output_path)
                
                    # for calc patients metric
                    video_metrics = self.calc()
                    self.update_report(patient_no, video_name, video_metrics, target='video')
                    
                    videos_metrics_list.append(video_metrics)
                    
                # calc each patients CR, OR
                patient_metrics = self.metric_helper.aggregate_calc_metric(videos_metrics_list)
                self.update_report(patient_no, video_name, patient_metrics, target='patient')
                
                # for calc total patients CR, OR
                patients_metrics_list.append(patient_metrics)
                

        self.update_report(patient_no, video_name, patients_metrics_list, target='all')
        
    def sort_for_patient(self, output_path_list):
        """
            patient_path_list
             - patient
                - result csv path list (per video)
        """
        patient_path_list = {}
        
        for output_path in output_path_list:
            tokens = output_path.split('/')
            patient = tokens[-2]
            video_name = tokens[-1][:-4]
            
            if patient in patient_path_list:
                patient_path_list[patient].append([video_name, output_path])
            else:
                patient_path_list[patient] = [video_name, output_path]
                
        return patient_path_list
          
    def save_dict_to_csv(self, results_dict, save_path):
        results_df = pd.DataFrame.from_dict([results_dict]) # dict to df            
        results_df = results_df.reset_index(drop=True)
        
        if os.path.exists(save_path):
            prev_results = pd.read_csv(save_path)
            prev_results.drop(['Unnamed: 0'], axis = 1, inplace = True) # to remove Unmaned : 0 colume

            results_df = pd.concat([prev_results, results_df], ignore_index=True, sort=False)
            
        results_df.to_csv(save_path, mode='w')
