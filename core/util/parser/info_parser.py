import re
import os


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
