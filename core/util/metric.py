import math
import numpy as np
import os
import pandas as pd
from pycm import *
import matplotlib.pyplot as plt


class MetricHelper():
    """
        Help metric computation.
    """
    def __init__(self, args):
        self.args = args
        self.EXCEPTION_NUM = -100
        self.IB_CLASS, self.OOB_CLASS = (0, 1)
        self.n_classes = self.args.n_classes
        self.classes = list(range(self.n_classes))
        self.target_metric = self.args.target_metric
        
        if 'loss' in self.target_metric:
            self.best_metric = math.inf
        else:
            self.best_metric = 0
            
        self.epoch = 0
        
        self.loss_dict = {
            'train': [],
            'valid': [],
        }
        self.tmp_loss_dict = {
            'train': [],
            'valid': [],
        }
                
        self.output_dict = {
            'pred': list(),
            'gt': list(),
        }
        
        self.results = np.zeros((self.args.max_epoch, 1))
        
    def update_epoch(self, epoch):
        self.epoch = epoch
        
    def write_preds(self, pred_list, gt_list):
        for pred, gt in zip(pred_list, gt_list):
            self.output_dict['pred'].append(pred.item())
            self.output_dict['gt'].append(gt.item())
        
    def write_loss(self, loss_val, state='train'):
        self.tmp_loss_dict[state].append(loss_val)
            
    def update_loss(self, state='train'):
        avg_loss = sum(self.tmp_loss_dict[state]) / len(self.tmp_loss_dict[state])
        self.tmp_loss_dict[state] = []
        self.loss_dict[state].append(avg_loss)
            
    def calc_metric(self):
        """
            task metric computation
        """
        gt_list, pred_list = self.output_dict['gt'], self.output_dict['pred']

        cm = ConfusionMatrix(gt_list, pred_list, classes=[0, 1])
        
        try: # for [1,1,1] [1,1,1] error solving
            cm.relabel(mapping={0:self.IB_CLASS, 1:self.OOB_CLASS}) # when [1,1,0] [0,1,0] return => cm.classes : [0, 1]
        except:
            cm.relabel(mapping={'0':self.IB_CLASS, '1':self.OOB_CLASS}) # when [1,1,1] [1,1,1] return => cm.classes : ['0', '1']
            
        metrics = {
            'TP': cm.TP[self.OOB_CLASS],
            'TN': cm.TN[self.OOB_CLASS],
            'FP': cm.FP[self.OOB_CLASS],
            'FN': cm.FN[self.OOB_CLASS],
            'Accuracy': cm.ACC[self.OOB_CLASS],
            'Precision': cm.PPV[self.OOB_CLASS],
            'Recall': cm.TPR[self.OOB_CLASS],
            'F1-Score': cm.F1[self.OOB_CLASS],
            'Jaccard': cm.J[self.OOB_CLASS],
        }

        # np casting for zero divide to inf
        # TP = np.float16(metrics['TP'])
        # FP = np.float16(metrics['FP'])
        # TN = np.float16(metrics['TN'])
        # FN = np.float16(metrics['FN'])

        TP = float(metrics['TP'])
        FP = float(metrics['FP'])
        TN = float(metrics['TN'])
        FN = float(metrics['FN'])

    

        metrics['CR'] = (TP - FP) / (FN + TP + FP) # 잘못예측한 OOB / predict OOB + 실제 OOB # Confidence Ratio
        metrics['OR'] = FP / (FN + TP + FP) # Over estimation ratio
        metrics['Mean_metric'] = (metrics['CR'] + (1-metrics['OR'])) / 2 # for train

        # Predict / GT CLASS elements num
        metrics['gt_IB']= gt_list.count(self.IB_CLASS)
        metrics['gt_OOB']= gt_list.count(self.OOB_CLASS)
        metrics['predict_IB']= pred_list.count(self.IB_CLASS)
        metrics['predict_OOB']= pred_list.count(self.OOB_CLASS)
        
        if len(self.loss_dict['valid']) > 0:
            metrics['val_loss'] = self.loss_dict['valid'][-1]
                
        # exception
        for k, v in metrics.items():
            if v == 'None': # ConfusionMetrix return
                metrics[k] = self.EXCEPTION_NUM
            elif np.isinf(v) or np.isnan(v): # numpy return
                metrics[k] = self.EXCEPTION_NUM
    
        # save accuracy
        self.results[self.epoch-1] = metrics['Mean_metric']
        self.output_dict['gt'] = []
        self.output_dict['pred'] = []
        
        return metrics
    
    def aggregate_calc_metric(self, metrics_list):
        advanced_metrics = {
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0,
            'Accuracy': self.EXCEPTION_NUM,
            'Precision': self.EXCEPTION_NUM,
            'Recall': self.EXCEPTION_NUM,
            'F1-Score': self.EXCEPTION_NUM,
            'Jaccard': self.EXCEPTION_NUM, # TO-DO should calc
            'gt_IB': 0,
            'gt_OOB': 0,
            'predict_IB': 0,
            'predict_OOB': 0,
        }

        # sum of TP/TN/FP/FN
        for metrics in metrics_list:
            advanced_metrics['TP'] += metrics['TP']
            advanced_metrics['TN'] += metrics['TN']
            advanced_metrics['FP'] += metrics['FP']
            advanced_metrics['FN'] += metrics['FN']
            
            # sum IB / OOB
            advanced_metrics['gt_IB'] += metrics['gt_IB']
            advanced_metrics['gt_OOB'] += metrics['gt_OOB']
            advanced_metrics['predict_IB'] += metrics['predict_IB']
            advanced_metrics['predict_OOB'] += metrics['predict_OOB']

        # np casting for zero divide to inf
        TP = np.float16(advanced_metrics['TP'])
        FP = np.float16(advanced_metrics['FP'])
        TN = np.float16(advanced_metrics['TN'])
        FN = np.float16(advanced_metrics['FN'])

        advanced_metrics['Accuracy'] = (TP + TN) / (TP + TN + FP + FN)
        advanced_metrics['Precision'] = TP / (TP + FP)
        advanced_metrics['Recall'] = TP / (TP + FN)
        advanced_metrics['F1-Score'] = 2 * ((advanced_metrics['Precision'] * advanced_metrics['Recall']) / (advanced_metrics['Precision'] + advanced_metrics['Recall']))

        advanced_metrics['CR'] = (TP - FP) / (FN + TP +FP) # 잘못예측한 OOB / predict OOB + 실제 OOB # Confidence Ratio
        advanced_metrics['OR'] = FP / (FN + TP + FP)  # Over estimation ratio
        advanced_metrics['Mean_metric'] = (advanced_metrics['CR'] + (1-advanced_metrics['OR'])) / 2 # for train

        # calc mCR, mOR
        advanced_metrics['mCR'] = np.mean([metrics['CR'] for metrics in metrics_list])
        advanced_metrics['mOR'] = np.mean([metrics['OR'] for metrics in metrics_list])

        # calc mPrecision, mRecall
        advanced_metrics['mPrecision'] = np.mean([metrics['Precision'] for metrics in metrics_list])
        advanced_metrics['mRecall'] = np.mean([metrics['Recall'] for metrics in metrics_list])

        # calc Jaccard index (https://neo4j.com/docs/graph-data-science/current/alpha-algorithms/jaccard/)
        advanced_metrics['Jaccard'] = np.float16(advanced_metrics['TP']) / np.float16(advanced_metrics['predict_OOB'] + advanced_metrics['gt_OOB'] - advanced_metrics['TP'])

        # exception
        for k, v in advanced_metrics.items():
            if v == 'None': # ConfusionMetrix return
                advanced_metrics[k] = self.EXCEPTION_NUM
            elif np.isinf(v) or np.isnan(v): # numpy return
                advanced_metrics[k] = self.EXCEPTION_NUM

        return advanced_metrics
    
    def get_best_metric(self):
        return self.best_metric
            
    def get_loss(self, state='train'):
        return self.loss_dict[state][-1]
            
    def save_loss_pic(self):
        fig = plt.figure(figsize=(32, 16))

        plt.ylabel('Loss', fontsize=50)
        plt.xlabel('Epoch', fontsize=50)
        
        plt.plot(range(self.epoch), self.loss_dict['train'])
        plt.plot(range(self.epoch), self.loss_dict['valid'])
        
        plt.legend(['Train', 'Val'], fontsize=40)
        plt.savefig(self.args.save_path + '/loss.png')
        
    def update_best_metric(self, metric):
        target_met = metric[self.target_metric]
        
        if 'loss' in self.target_metric:    
            if self.best_metric > target_met:
                self.best_metric = target_met
                return True
        else:
            if self.best_metric < target_met:
                self.best_metric = target_met
                return True
        
        return False
        