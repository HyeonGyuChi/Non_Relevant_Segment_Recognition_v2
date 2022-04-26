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


import sys


def main(target_frame_base_path, target_anno_base_path, time_th, ssim_score_th):
    from core.util.ssim import SSIM
    from core.util.ssim import get_anno_list,convert_json_path_to_video_path,get_video_meta_info_from_ffmpeg,fps_tuning
    import json
    import numpy as np
    import pandas as pd
    from config.base_opts import parse_opts
    parser = parse_opts()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    n_cpu = 60

    ray.init(num_cpus=n_cpu)

    target_anno_list = glob.glob(os.path.join(target_anno_base_path, '*.json'))
    target_anno_list = natsort.natsorted(target_anno_list)

    patients_dict = defaultdict(list)
    for target_anno in target_anno_list:
        patients_dict['_'.join(target_anno.split('/')[-1].split('_')[:5])].append(target_anno)
    patients_list = list(patients_dict.values())

    for patient in patients_list: 
        #### 이상치 비디오 예외 처리 (3건) - 04_GS4_99_L_47, 01_VIHUB1.2_A9_L_33, 01_VIHUB1.2_B4_L_79
        if '_'.join(patient[0].split('/')[-1].split('_')[:5]) in ['04_GS4_99_L_47', '01_VIHUB1.2_A9_L_33', '01_VIHUB1.2_B4_L_79']:
            continue

        per_patient_list = []

        print("patient",patient)
        for target_anno in patient: 
            print('\n+[target anno]  : {}'.format(target_anno))

            patient_full_name = '_'.join(target_anno.split('/')[-1].split('_')[:5]) # 01_VIHUB1.2_A9_L_2
            video_no = target_anno.split('/')[-1][:-5]

            print('+[target frames] : {}'.format(os.path.join(target_frame_base_path, patient_full_name, video_no)))
            target_frames = glob.glob(os.path.join(target_frame_base_path, patient_full_name, video_no, '*.jpg'))
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
            time_idx = [SSIM(args).idx_to_time(idx, 30) for idx in frame_idx]
            
    
            patient_df['frame_idx'] = frame_idx
            patient_df['time_idx'] = time_idx
            patient_df = patient_df[['frame_idx', 'time_idx', 'frame_path', 'gt', 'ssim']]
            print('\n\n\t\t<< patient df >>\n')
            print(patient_df)


            # NRS 이미지만 ssim 계산
            patient_df_copy =  pd.DataFrame()
            json_path = target_anno

            with open(json_path, 'r') as f:
                json_data = json.load(f)
                for i in range(len(json_data['annotations'])):
                    start = patient_df['frame_idx'] >= json_data['annotations'][i]["start"]
                    end = patient_df['frame_idx'] <= json_data['annotations'][i]["end"]
                    subset_df = patient_df[start & end]              
                    patient_df_copy=pd.concat([patient_df_copy, subset_df], axis = 0)
            # print('\n\n\t\t<< patient df_NRS >>\n')
            # print(patient_df_copy)


            # ######### calculate ssim score #########
            final_df_1_fps = SSIM(args).compute_ssim(target_assets_df=patient_df_copy, ssim_score_th=ssim_score_th, n_cpu=10)
            print('\n\n\t\t<< final_df_1_fps >>\n')
            print(final_df_1_fps) 

            # final_df_5_fps = SSIM.compute_ssim(target_assets_df=patient_df, ssim_score_th=ssim_score_th, n_cpu=50)
            # print('\n\n\t\t<< final_df_5_fps >>\n')
            # print("final_df_1_fps",final_df_5_fps) 
            
            print('\n\n\t\t<< JSON 수정 >>')
            print(target_anno)
            from core.util.anno2json import Anno2Json
            annotation_to_json = Anno2Json(args,final_df_1_fps,target_anno)
            annotation_to_json.make_json(version="ssim")
            print()
            print()
            print()
        

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

    main(target_frame_base_path = '../core/dataset/NRS/ssim/lapa/vihub/img/', 
    target_anno_base_path = '../core/dataset/NRS/ssim/lapa/vihub/anno/', 
    time_th=0, 
    ssim_score_th=0.997)

    ed = time.time()
    elapsed_time = ed-st
    print('{:.6f} seconds'.format(elapsed_time))


