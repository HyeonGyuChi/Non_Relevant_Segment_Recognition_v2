import os
import json
import pandas as pd
import numpy as np
import yaml
import re

class AnnotationParser():
    def __init__(self, annotation_path:str):
        self.IB_CLASS, self.OOB_CLASS = (0,1)
        self.annotation_path = annotation_path
        print('annotation_path : ', annotation_path)
        self.json_data = FileLoader(annotation_path).load()
    
    def set_annotation_path(self, annotation_path):
        self.annotation_path = annotation_path
        self.json_data = FileLoader(annotation_path).load()

    def get_totalFrame(self):
        return self.json_data['totalFrame'] # str

    def get_fps(self):
        return self.json_data['frameRate'] # float
    
    def get_annotations_info(self):
        # annotation frame
        annotation_idx_list = []
        
        for anno_data in self.json_data['annotations'] :
            start = anno_data['start'] # start frame number
            end = anno_data['end'] # end frame number

            annotation_idx_list.append([start, end]) # annotation_idx_list = [[start, end], [start, end]..]

        return annotation_idx_list

    def get_event_sequence(self, extract_interval=1):

        event_sequence = np.array([self.IB_CLASS]*self.get_totalFrame())
        annotation_idx_list = self.get_annotations_info()
        
        for start, end in annotation_idx_list:
            event_sequence[start: end+1] = self.OOB_CLASS
        
        return event_sequence.tolist()[::extract_interval]

class FileLoader():
    def __init__(self, file_path=''):
        self.file_path = file_path

    def set_file_path(self, file_path):
        self.file_path = file_path

    def get_full_path(self):
        return os.path.abspath(self.file_path)

    def get_file_name(self):
        return os.path.splitext(self.get_basename())[0]

    def get_file_ext(self):
        return os.path.splitext(self.get_basename())[1]
    
    def get_basename(self):
        return os.path.basename(self.file_path)
    
    def get_dirname(self):
        return os.path.dirname(self.file_path)
    
    def load(self):
        # https://stackoverflow.com/questions/9168340/using-a-dictionary-to-select-function-to-execute
        support_loader = {
            '.json':(lambda x: self.load_json()), # json object
            '.csv':(lambda x: self.load_csv()), # Dataframe
            '.yaml':(lambda x: self.load_yaml()), # dict
            '.png':-1 # PIL
        }

        data = support_loader.get(self.get_file_ext(), -1)('dummy')

        # assert data_loader != -1, 'NOT SUPPOERT FILE EXTENSION ON FileLoader'

        return data

    def load_json(self): # to json object
        with open(self.file_path) as self.json_file :
            return json.load(self.json_file)

    def load_csv(self): # to Dataframe
        df = pd.read_csv(self.file_path)
        return df
    
    def load_yaml(self): # to dict
        load_dict = {}

        with open(self.file_path, 'r') as f :
            load_dict = yaml.load(f, Loader=yaml.FullLoader)
    
        return load_dict


class InfoParser():
    def __init__(self, parser_type='ROBOT_VIDEO_1'):
        self.file_name = '' # name or path
        self.parser_type = parser_type

    def write_file_name(self, file_name):
        self.file_name = file_name

    def get_info(self):
        info = {
            'hospital':'',
            'surgery_type':'',
            'surgeon':'',
            'op_method':'',
            'patient_idx':'',
            'video_channel':'',
            'video_slice_no':''
        }
        
        support_parser = { 
            'ROBOT_VIDEO_1': (lambda x: self._robot_video_name_to_info_v1()), # for Robot video 40
            'ROBOT_VIDEO_2':(lambda x: self._robot_video_name_to_info_v2()), # for Robot video 60
            'LAPA_VIDEO_1':(lambda x: self._lapa_video_name_to_info_v1()), # for Lapa video 100
            'ROBOT_ANNOTATION':(lambda x: self._robot_annotation_name_to_info()), # for Robot annotation,
            'ETC_VIDEO_1': (lambda x: self._etc_video_name_to_info()), # for etc video 24
            'ETC_ANNOTATION': (lambda x: self._etc_annotation_name_to_info()), # for etc annotation 24
        }

        # return rule
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no = support_parser.get(self.parser_type, -1)('dummy')

        info = {
            'hospital':hospital,
            'surgery_type':surgery_type,
            'surgeon':surgeon,
            'op_method':op_method,
            'patient_idx':patient_idx,
            'video_channel':video_channel,
            'video_slice_no':video_slice_no
        }

        return info

    def get_video_name(self): # R_310_ch1_01
        info = self.get_info()
        video_name = [info['op_method'], info['patient_idx'], info['video_channel'], info['video_slice_no']]
        return '_'.join(video_name)

    def get_patient_no(self): # R_310
        info = self.get_info()
        patient_no = [info['op_method'], info['patient_idx']]
        return '_'.join(patient_no)

    def _clean_file_ext(self): # /data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2/01_G_01_R_100_ch1_01.mp4' => 01_G_01_R_100_ch1_01
        return os.path.splitext(os.path.basename(self.file_name))[0]
        

    def _robot_video_name_to_info_v1(self):
        parents_dir, video = self.file_name.split(os.path.sep)[-2:] # R000001, ch1_video_01.mp4
        video_name, ext = os.path.splitext(video) # ch1_video_01, .mp4

        op_method, patient_idx = re.findall(r'R|\d+', parents_dir) # R, 000001
        
        patient_idx = str(int(patient_idx)) # 000001 => 1

        video_channel, _, video_slice_no = video_name.split('_') # ch1, video, 01

        new_nas_policy_name = "_".join([op_method, patient_idx, video_channel, video_slice_no]) # R_1_ch1_01
        # print('CONVERTED NAMING: {} \t ===> \t {}'.format(self.file_name, new_nas_policy_name))

        hospital = '01'
        surgery_type = 'G'
        surgeon = '01'

        return hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no

    def _robot_video_name_to_info_v2(self):
        file_name = self._clean_file_ext()
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no = file_name.split('_') # parsing video name
    
    def _lapa_video_name_to_info_v1(self):
        file_name = self._clean_file_ext()
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no = file_name.split('_') # parsing video name

        return hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no

    def _robot_annotation_name_to_info(self):
        file_name = self._clean_file_ext()
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no, _, _ = file_name.split('_') # parsing annotation name

        return hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no

    def _etc_video_name_to_info(self):
        file_name = self._clean_file_ext()
        hospital, surgery_type, surgeon, op_method, patient_idx, video_slice_no, = file_name.split('_') # parsing video name
        video_channel = 'empty'

        return hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no

    def _etc_annotation_name_to_info(self):
        file_name = self._clean_file_ext()
        hospital, surgery_type, surgeon, op_method, patient_idx, video_slice_no, _, _ = file_name.split('_') # parsing annotation name
        video_channel = 'empty'

        return hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no
