import json
import numpy as np


class Reporter():
    def __init__(self):
        self.report_path = None
        self.reset()
    
    def set_save_path(self, report_path):
        self.report_path = report_path
        
    def reset(self):
        self.total_report = self._init_total_report_form()
        self.experiment = self.total_report['experiment']
        self.patients_report = self.total_report['experiment']['patients'] # []
        self.videos_report = self.total_report['experiment']['videos'] # []
        
    def _init_total_report_form(self):
        init_total_report_form = {
            'experiment': {
                'model':'',
                'method':'',
                'inference_fold':'',
                'mCR':0,
                'mOR':0,
                'CR':0,
                'OR':0,
                'mPrecision':0,
                'mRecall':0,
                'Jaccard':0,
                'details_path':'',
                'model_path':'',

                'patients':[],
                'videos':[],
            },
        }

        return init_total_report_form
    
    def _get_report_form(self, report_type):
        init_report_form = { # one-columne of experiments report
            'patient': { # one-columne of inference report (each patients)
                'patient_no' : '',
                'FP' : 0,
                'TP' : 0,
                'FN' : 0,
                'TN' : 0,
                'TOTAL' : 0,
                'CR' : 0,
                'OR' : 0,
                'Precision': 0,
                'Recall': 0,
                'Jaccard':0,
                'gt_IB':0,
                'gt_OOB':0,
                'predict_IB':0,
                'predict_OOB':0,
            },

            'video': { # one-columne of inference report (each videos)
                'patient_no' : '',
                'video_no' : '',
                'FP' : 0,
                'TP' : 0,
                'FN' : 0,
                'TN' : 0,
                'TOTAL' : 0,
                'CR' : 0,
                'OR' : 0,
                'Precision': 0,
                'Recall': 0,
                'Jaccard':0,
                'gt_IB':0,
                'gt_OOB':0,
                'predict_IB':0,
                'predict_OOB':0,
            }
        }

        return init_report_form[report_type]
    
    def set_experiment(self, model, methods, inference_fold, 
                       mCR, mOR, CR, OR, 
                       mPrecision, mRecall, Jaccard, 
                       details_path, model_path):
        
        self.experiment['model'] = model
        self.experiment['method'] = methods
        self.experiment['inference_fold'] = inference_fold
        self.experiment['mCR'] = mCR
        self.experiment['mOR'] = mOR
        self.experiment['CR'] = CR
        self.experiment['OR'] = OR
        self.experiment['mPrecision'] = mPrecision
        self.experiment['mRecall'] = mRecall
        self.experiment['Jaccard'] = Jaccard
        # TODO - patients.. 
        self.experiment['details_path'] = details_path
        self.experiment['model_path'] = model_path
    
    def add_patients_report(self, patient_no, 
                            FP, TP, FN, TN, TOTAL, 
                            CR, OR, gt_IB, gt_OOB, 
                            predict_IB, predict_OOB, 
                            precision, recall, jaccard):
        
        patient = self._get_report_form('patient')

        patient['patient_no'] = patient_no
        patient['FP'] = FP
        patient['TP'] = TP
        patient['FN'] = FN
        patient['TN'] = TN
        patient['TOTAL'] = TOTAL
        patient['CR'] = CR
        patient['OR'] = OR
        patient['gt_IB'] = gt_IB
        patient['gt_OOB'] = gt_OOB
        patient['predict_IB'] = predict_IB
        patient['predict_OOB'] = predict_OOB

        patient['Precision'] = precision
        patient['Recall'] = recall

        patient['Jaccard'] = jaccard
        
        self.patients_report.append(patient)

        return patient
    
    def add_videos_report(self, patient_no, video_no, 
                          FP, TP, FN, TN, TOTAL, 
                          CR, OR, gt_IB, gt_OOB, 
                          predict_IB, predict_OOB, 
                          precision, recall, jaccard):
        
        video = self._get_report_form('video')

        video['patient_no'] = patient_no
        video['video_no'] = video_no
        video['FP'] = FP
        video['TP'] = TP
        video['FN'] = FN
        video['TN'] = TN
        video['TOTAL'] = TOTAL
        video['CR'] = CR
        video['OR'] = OR
        video['gt_IB'] = gt_IB
        video['gt_OOB'] = gt_OOB
        video['predict_IB'] = predict_IB
        video['predict_OOB'] = predict_OOB

        video['Precision'] = precision
        video['Recall'] = recall

        video['Jaccard'] = jaccard
        
        self.videos_report.append(video)
        
        return video
    
    def save_report(self):
        print('report path : ', self.report_path)
        json_string = json.dumps(self.total_report, indent=4, cls=MyEncoder)
        print(json_string)

        with open(self.report_path, "w") as json_file:
            json.dump(self.total_report, json_file, indent=4, cls=MyEncoder)
            
            
            
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)