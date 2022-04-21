import cv2
import ray
import time
import imutils
import datetime

from skimage.metrics import structural_similarity as ssim

import os
import glob
import natsort
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby

import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

import datetime

# from visualization import visual_sampling
# from report import report_per_video, report_per_patient
# from gif import gen_gif

import sys


def main(target_frame_base_path, target_anno_base_path, time_th, ssim_score_th):
    from core.util.ssim import SSIM
    # target_frame_base_path = '../core/dataset/NRS/ssim/lapa/vihub/01_VIHUB1.2_A9_L_15/01_VIHUB1.2_A9_L_15_01/img', target_anno_base_path = '../core/dataset/NRS/ssim/lapa/vihub/01_VIHUB1.2_A9_L_15/01_VIHUB1.2_A9_L_15_01/anno/',
    
    # pd.set_option('display.max_rows', None)
    n_cpu = 60

    ray.init(num_cpus=n_cpu)

    target_anno_list = glob.glob(os.path.join(target_anno_base_path, '*.json'))
    target_anno_list = natsort.natsorted(target_anno_list)

    patients_dict = defaultdict(list)
    for target_anno in target_anno_list:
        patients_dict['_'.join(target_anno.split('/')[-1].split('_')[:5])].append(target_anno)

    patients_list = list(patients_dict.values())

    print("target_anno_list",target_anno_list)
    print("patients_list",patients_list)
    exit()

    for patient in patients_list: 
        #### 이상치 비디오 예외 처리 (3건) - 04_GS4_99_L_47, 01_VIHUB1.2_A9_L_33, 01_VIHUB1.2_B4_L_79
        if '_'.join(patient[0].split('/')[-1].split('_')[:5]) in ['04_GS4_99_L_47', '01_VIHUB1.2_A9_L_33', '01_VIHUB1.2_B4_L_79']:
            continue

        per_patient_list = []

        for target_anno in patient: 
            
            # ['/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS/04_GS4_99_L_1_01_NRS_30.json', '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS/04_GS4_99_L_1_02_NRS_30.json']
            '''
            target_anno = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS/04_GS4_99_L_1_02_NRS_30.json'
            target_frames = '/raid/img_db/VIHUB/gangbuksamsung_127case/L_1/04_GS4_99_L_1_02'
            /data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_1st/NRS/01_VIHUB1.2_A9_L_2_01_NRS_12.json
            /raid/img_db/VIHUB/severance_1st/01_VIHUB1.2_A9/L_1/01_VIHUB1.2_A9_L_1_01
            '''
            print('\n+[target anno] : {}'.format(target_anno))

            patient_full_name = '_'.join(target_anno.split('/')[-1].split('_')[:5]) # 01_VIHUB1.2_A9_L_2
            patient_no = '_'.join(target_anno.split('/')[-1].split('_')[3:5]) # L_2
            
            video_no = '_'.join(target_anno.split('/')[-1].split('_')[:6]) # 1_VIHUB1.2_A9_L_2_01

            ## target_frames | get frames from img_db (path 설정)
            if 'gangbuksamsung_127case' in target_anno.split('/'):
                print('+[target frames] : {}'.format(os.path.join(target_frame_base_path, patient_no, video_no)))
                target_frames = glob.glob(os.path.join(target_frame_base_path, patient_no, video_no, '*.jpg'))
                target_frames = natsort.natsorted(target_frames)

            elif ('severance_1st' in target_anno.split('/')) or ('severance_2nd' in target_anno.split('/')):
                severance_path = '_'.join(target_anno.split('/')[-1].split('_')[:3]) # 22.01.24 jh 추가

                print('+[target frames] : {}'.format(os.path.join(target_frame_base_path, severance_path, patient_no, video_no)))
                target_frames = glob.glob(os.path.join(target_frame_base_path, severance_path, patient_no, video_no, '*.jpg'))
                target_frames = natsort.natsorted(target_frames)

            gt_list, ssim_list = get_anno_list(target_anno=target_anno, total_len=len(target_frames), time_th=time_th)
            
            assets_data = {
                'frame_path': target_frames,
                'gt' : gt_list,
                'ssim' : ssim_list
            }
            assets_df = pd.DataFrame(assets_data)

            per_patient_list.append(assets_df)

        ######## df per patient (30fps) ########
        # TODO 추출한 frame 수와 totalFrame 수 unmatch
        patient_df = pd.concat(per_patient_list, ignore_index=True)

        # frame_idx, time_idx 추가
        frame_idx = list(range(0, len(patient_df)))
        time_idx = [idx_to_time(idx, fps=30) for idx in frame_idx]

        patient_df['frame_idx'] = frame_idx
        patient_df['time_idx'] = time_idx

        patient_df = patient_df[['frame_idx', 'time_idx', 'frame_path', 'gt', 'ssim']]

        print('\n\n\t\t<< patient df >>\n')
        print(patient_df)

        ######### get video origin FPS #########
        print(patient[0])
        video_path = SSIM.convert_json_path_to_video_path(patient[0])
        VIDEO_FPS = SSIM.get_video_meta_info_from_ffmpeg(video_path)

        print(VIDEO_FPS)
        
        if VIDEO_FPS >= 29.0 and VIDEO_FPS <=31.0:
            VIDEO_FPS = 30
        elif VIDEO_FPS >= 59.0 and VIDEO_FPS <=61.0:
            VIDEO_FPS = 60

        # ########## fps tuning (1fps, 5fps) ##########
        patient_df_1_fps = SSIM.fps_tuning(target_assets_df=patient_df, target_fps=1, VIDEO_FPS=VIDEO_FPS)
        patient_df_5_fps = SSIM.fps_tuning(target_assets_df=patient_df, target_fps=5, VIDEO_FPS=VIDEO_FPS)

        print('\n\n\t\t<< patient_df_1_fps >>\n')
        print(patient_df_1_fps)

        print('\n\n\t\t<< patient_df_5_fps >>\n')
        print(patient_df_5_fps)

         # ######### calculate ssim score #########
        final_df_1_fps = SSIM.compute_ssim(patient_df_1_fps, ssim_score_th, n_cpu=10)
        final_df_5_fps = SSIM.compute_ssim(patient_df_5_fps, ssim_score_th, n_cpu=50)

        print('\n\n\t\t<< final_df_1_fps >>\n')
        print(final_df_1_fps)

        print('\n\n\t\t<< final_df_5_fps >>\n')
        print(final_df_5_fps)


        base_save_path = '/raid/SSIM_RESULT/{}-SSIM_RESULT'.format(ssim_score_th)
        ################ save df #################
        df_save_path = os.path.join(base_save_path, target_anno.split('/')[-3], patient_full_name)
        os.makedirs(df_save_path, exist_ok=True)
        final_df_1_fps.to_csv(os.path.join(df_save_path, '{}-1FPS.csv'.format(patient_full_name)))
        final_df_5_fps.to_csv(os.path.join(df_save_path, '{}-5FPS.csv'.format(patient_full_name)))

        ############# video report ###############
        report_per_video_save_path = os.path.join(base_save_path, target_anno.split('/')[-3], patient_full_name) # ssim_result/gangbuksamsung_127case/04_GS4_99_L_1
        report_per_video(assets_df=final_df_1_fps, target_fps='1', time_th=time_th, ssim_score_th=ssim_score_th, save_path=report_per_video_save_path)
        report_per_video(assets_df=final_df_5_fps, target_fps='5', time_th=time_th, ssim_score_th=ssim_score_th, save_path=report_per_video_save_path)

        ############ patient report ##############
        patient_name = '_'.join(patient[0].split('/')[-1].split('_')[:5])
        report_per_patient_save_path = os.path.join(base_save_path, target_anno.split('/')[-3]) # ssim_result/gangbuksamsung_127case
        report_per_patient(assets_df=final_df_1_fps, patient_name=patient_name, target_fps='1', time_th=time_th, ssim_score_th=ssim_score_th, save_path=report_per_patient_save_path, VIDEO_FPS=VIDEO_FPS)
        report_per_patient(assets_df=final_df_5_fps, patient_name=patient_name, target_fps='5', time_th=time_th, ssim_score_th=ssim_score_th, save_path=report_per_patient_save_path, VIDEO_FPS=VIDEO_FPS)
        
        ############## visualization #############
        visual_save_path = os.path.join(base_save_path, target_anno.split('/')[-3], patient_full_name) # ssim_result/gangbuksamsung_127case/04_GS4_99_L_1
        visual_sampling(final_df_1_fps, final_df_5_fps, window_size=5000, patient_no=patient_no, time_th=time_th, ssim_score_th=ssim_score_th, save_path=visual_save_path, VIDEO_FPS=VIDEO_FPS)
        
        ################### gif ##################
        gif_save_path = os.path.join(base_save_path, target_anno.split('/')[-3], patient_full_name, 'gif') # ssim_result/gangbuksamsung_127case/04_GS4_99_L_1/gif
        gen_gif(assets_df=final_df_1_fps, target_fps=1, save_path=gif_save_path) 
        gen_gif(assets_df=final_df_5_fps, target_fps=5, save_path=gif_save_path) 

    ray.shutdown()


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        # __file__ = /home/jiwoo/NRS/script/ssim.py
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        print(base_path)
    
    st = time.time()

    # sample: /raid/NRS/ssim/lapa/vihub/01_VIHUB1.2_A9_L_15/01_VIHUB1.2_A9_L_15_01
    # json으로 NRS part - image match
    # ssim
    # json rewrite
    # csv write
    main(target_frame_base_path = '../core/dataset/NRS/ssim/lapa/vihub/01_VIHUB1.2_A9_L_15/01_VIHUB1.2_A9_L_15_01/img', target_anno_base_path = '../core/dataset/NRS/ssim/lapa/vihub/01_VIHUB1.2_A9_L_15/01_VIHUB1.2_A9_L_15_01/anno/', time_th=0, ssim_score_th=0.997)

    ed = time.time()
    elapsed_time = ed-st
    print('{:.6f} seconds'.format(elapsed_time))






