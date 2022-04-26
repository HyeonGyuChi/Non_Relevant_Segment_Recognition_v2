import cv2
import ray
import time
import imutils
import datetime

from skimage.metrics import structural_similarity as ssim
# from core.util.parser import AssetParser



class SSIM():
    def __init__(self, args):
        self.args = args
        
        if self.args.dataset == 'robot':
            if self.args.datatype == 'vihub':
                pass
            elif self.args.datatype == 'etc':
                pass
            elif self.args.datatype == 'mola':
                self.patients_list = mola[state][self.args.fold]
        elif self.args.dataset == 'lapa':
            pass
           
        
        self.img_base_path = self.args.data_base_path + '/{}/{}/img'.format(self.args.dataset,
                                                                                self.args.datatype)
        
        self.anno_path = self.args.data_base_path + '/{}/{}/anno/{}'.format(self.args.dataset,
                                                                                self.args.datatype,
                                                                                self.args.data_version)
        self.save_anno_path = self.args.data_base_path + '/{}/{}/anno/{}_ssim'.format(self.args.dataset,
                                                                                self.args.datatype,
                                                                                self.args.data_version)
        
    
    
    @ray.remote
    def cal_ssim_score(target_ssim_list, st_idx, ed_idx):
        ssim_score_list = []
        import numpy as np
        import cv2
        
        print("======calculate ssim score=====")
        for i in range(st_idx, ed_idx):
            prev_path, cur_path = target_ssim_list[i], target_ssim_list[i+1]

            # cal_ssim() ERROR: This function was not imported properly. python
            # Load the two input images
            imageA = cv2.imread(prev_path)
            imageB = cv2.imread(cur_path)

            # Convert the images to grayscale
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

            # Compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            score, _ = ssim(grayA, grayB, full=True)
            ssim_score_list.append(score)
            
        return ssim_score_list
    

    def compute_ssim(self,target_assets_df, ssim_score_th, n_cpu):
        import numpy as np
        import pandas as pd
        # ray :)
        target_ssim_df = target_assets_df[target_assets_df.ssim == 1][['frame_path']]
        target_ssim_list = target_ssim_df['frame_path'].values.tolist()

        split = len(target_ssim_list) // n_cpu
        st_list = [0 + split*i for i in range(n_cpu)]
        ed_list = [split*(i+1) for i in range(n_cpu)]
        ed_list[-1] = len(target_ssim_list)-1

        ray_target_ssim_list = ray.put(target_ssim_list)
        results = ray.get([SSIM.cal_ssim_score.remote(ray_target_ssim_list, st_list[i], ed_list[i]) for i in range(n_cpu)])

        ssim_score_list = []
        for res in results:
            ssim_score_list += res

        ssim_score_list.append(-100)

        # TODO
        target_ssim_df['ssim_score'] = ssim_score_list

        target_assets_df = pd.merge(target_assets_df, target_ssim_df, on='frame_path', how='outer')
        target_assets_df = target_assets_df.fillna(-1) # Nan -> -1

        condition_list = [
            ((target_assets_df['gt'] == 0) & (target_assets_df['ssim_score'] < ssim_score_th)), # RS & non-duplicate
            ((target_assets_df['gt'] == 0) & (target_assets_df['ssim_score'] >= ssim_score_th)), # RS & duplicate
            ((target_assets_df['gt'] == 1) & (target_assets_df['ssim_score'] < ssim_score_th)), # NRS & non-duplicate
            ((target_assets_df['gt'] == 1) & (target_assets_df['ssim_score'] >= ssim_score_th)) # NRS & duplicate
        ]

        choice_list = [0, 1, 2, 3]
        target_assets_df['class'] = np.select(condition_list, choice_list, default=-1)
        return target_assets_df


    def cal_ssim(self,img_1, img_2):
        # Load the two input images
        imageA = cv2.imread(img_1)
        imageB = cv2.imread(img_2)

        # Convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # Compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        score, _ = ssim(grayA, grayB, full=True)
        
        return score

    def idx_to_time(args,idx, fps):
        time_s = idx // fps
        frame = int(idx % fps)

        converted_time = str(datetime.timedelta(seconds=time_s))
        converted_time = converted_time + ':' + str(frame)

        return converted_time



def get_anno_list(target_anno, total_len, time_th):
    import os
    import json
    import numpy as np
    # select target annotation list (30초 이상 & nrs 인 경우)
    gt_chunk_list = [] # [[1, 100], [59723, 61008], [67650, 72319]]
    ssim_chunk_list = [] # [[59723, 61008], [67650, 72319]]
    
    gt_list = [0] * total_len # [0,1,1,1,1, ..., 1,0,0]
    ssim_list = [1] * total_len # [0,0,0,0,0, ..., 1,0,0] # 전체 프레임에 대해 검사 - 22.01.27 JH 추가

    with open(target_anno, 'r') as json_f:
        json_data = json.load(json_f)

    for json_f in json_data['annotations']:
        gt_chunk_list.append([json_f['start'], json_f['end']]) # [[1, 100], [59723, 61008], [67650, 72319]]ß

    gt_np = np.array(gt_list)
    for gt_chunk in gt_chunk_list:
        gt_np[gt_chunk[0]:gt_chunk[1]] = 1        
    
    gt_list = gt_np.tolist()
    return gt_list, ssim_list


def convert_json_path_to_video_path(json_path):
    import glob
    import os
    # json_path = "../core/dataset/NRS/ssim/lapa/vihub/anno/01_VIHUB1.2_A9_L_15_01.json"
    video_base_path = '../core/dataset/NRS/ssim/lapa/vihub/img'

    patient_name = '_'.join(json_path.split('/')[-1].split('_')[:5])
    video_name = ('_'.join(json_path.split('/')[-1].split('_')[:6]))[:-5]
    video_path = glob.glob(os.path.join(video_base_path, patient_name, video_name, '*.jpg'))

    
    return video_path[0]


def get_video_meta_info_from_ffmpeg(video_path):

    print('\n\n \t\t <<<<< GET META INFO FROM FFMPEG >>>>> \t\t \n\n')

    ffmpeg_helper = ffmpegHelper(video_path)
    fps = ffmpeg_helper.get_video_fps()

    return fps


def fps_tuning(target_assets_df, target_fps, VIDEO_FPS):
    interval = VIDEO_FPS // target_fps # TODO target video 의 fps 읽어와야 함. 현재는 30 fps 라고 고정하고 진행. 
    interval_list = list(range(0, len(target_assets_df), interval))

    fps_tuning_df = target_assets_df.loc[interval_list]

    return fps_tuning_df







# def main(target_frame_base_path, target_anno_base_path, time_th, ssim_score_th):
#     # pd.set_option('display.max_rows', None)
#     n_cpu = 60

#     ray.init(num_cpus=n_cpu)

#     target_anno_list = glob.glob(os.path.join(target_anno_base_path, '*.json'))
#     target_anno_list = natsort.natsorted(target_anno_list)
 
#     patients_dict = defaultdict(list)
#     for target_anno in target_anno_list:
#         patients_dict['_'.join(target_anno.split('/')[-1].split('_')[:5])].append(target_anno)

#     patients_list = list(patients_dict.values())

#     for patient in patients_list: 
#         #### 이상치 비디오 예외 처리 (3건) - 04_GS4_99_L_47, 01_VIHUB1.2_A9_L_33, 01_VIHUB1.2_B4_L_79
#         if '_'.join(patient[0].split('/')[-1].split('_')[:5]) in ['04_GS4_99_L_47', '01_VIHUB1.2_A9_L_33', '01_VIHUB1.2_B4_L_79']:
#             continue

#         per_patient_list = []

#         for target_anno in patient: 
#             # ['/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS/04_GS4_99_L_1_01_NRS_30.json', '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS/04_GS4_99_L_1_02_NRS_30.json']
#             '''
#             target_anno = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS/04_GS4_99_L_1_02_NRS_30.json'
#             target_frames = '/raid/img_db/VIHUB/gangbuksamsung_127case/L_1/04_GS4_99_L_1_02'
#             /data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_1st/NRS/01_VIHUB1.2_A9_L_2_01_NRS_12.json
#             /raid/img_db/VIHUB/severance_1st/01_VIHUB1.2_A9/L_1/01_VIHUB1.2_A9_L_1_01
#             '''
#             print('\n+[target anno] : {}'.format(target_anno))

#             patient_full_name = '_'.join(target_anno.split('/')[-1].split('_')[:5]) # 01_VIHUB1.2_A9_L_2
#             patient_no = '_'.join(target_anno.split('/')[-1].split('_')[3:5]) # L_2
            
#             video_no = '_'.join(target_anno.split('/')[-1].split('_')[:6]) # 1_VIHUB1.2_A9_L_2_01

#             ## target_frames | get frames from img_db (path 설정)
#             if 'gangbuksamsung_127case' in target_anno.split('/'):
#                 print('+[target frames] : {}'.format(os.path.join(target_frame_base_path, patient_no, video_no)))
#                 target_frames = glob.glob(os.path.join(target_frame_base_path, patient_no, video_no, '*.jpg'))
#                 target_frames = natsort.natsorted(target_frames)

#             elif ('severance_1st' in target_anno.split('/')) or ('severance_2nd' in target_anno.split('/')):
#                 severance_path = '_'.join(target_anno.split('/')[-1].split('_')[:3]) # 22.01.24 jh 추가

#                 print('+[target frames] : {}'.format(os.path.join(target_frame_base_path, severance_path, patient_no, video_no)))
#                 target_frames = glob.glob(os.path.join(target_frame_base_path, severance_path, patient_no, video_no, '*.jpg'))
#                 target_frames = natsort.natsorted(target_frames)

#             gt_list, ssim_list = get_anno_list(target_anno=target_anno, total_len=len(target_frames), time_th=time_th)
            
#             assets_data = {
#                 'frame_path': target_frames,
#                 'gt' : gt_list,
#                 'ssim' : ssim_list
#             }
#             assets_df = pd.DataFrame(assets_data)

#             per_patient_list.append(assets_df)

#         ######## df per patient (30fps) ########
#         # TODO 추출한 frame 수와 totalFrame 수 unmatch
#         patient_df = pd.concat(per_patient_list, ignore_index=True)

#         # frame_idx, time_idx 추가
#         frame_idx = list(range(0, len(patient_df)))
#         time_idx = [idx_to_time(idx, fps=30) for idx in frame_idx]

#         patient_df['frame_idx'] = frame_idx
#         patient_df['time_idx'] = time_idx

#         patient_df = patient_df[['frame_idx', 'time_idx', 'frame_path', 'gt', 'ssim']]

#         print('\n\n\t\t<< patient df >>\n')
#         print(patient_df)

#         ######### get video origin FPS #########
#         print(patient[0])
#         video_path = convert_json_path_to_video_path(patient[0])
#         VIDEO_FPS = get_video_meta_info_from_ffmpeg(video_path)

#         print(VIDEO_FPS)
        
#         if VIDEO_FPS >= 29.0 and VIDEO_FPS <=31.0:
#             VIDEO_FPS = 30
#         elif VIDEO_FPS >= 59.0 and VIDEO_FPS <=61.0:
#             VIDEO_FPS = 60

#         # ########## fps tuning (1fps, 5fps) ##########
#         patient_df_1_fps = fps_tuning(target_assets_df=patient_df, target_fps=1, VIDEO_FPS=VIDEO_FPS)
#         patient_df_5_fps = fps_tuning(target_assets_df=patient_df, target_fps=5, VIDEO_FPS=VIDEO_FPS)

#         print('\n\n\t\t<< patient_df_1_fps >>\n')
#         print(patient_df_1_fps)

#         print('\n\n\t\t<< patient_df_5_fps >>\n')
#         print(patient_df_5_fps)

#          # ######### calculate ssim score #########
#         final_df_1_fps = compute_ssim(patient_df_1_fps, ssim_score_th, n_cpu=10)
#         final_df_5_fps = compute_ssim(patient_df_5_fps, ssim_score_th, n_cpu=50)

#         print('\n\n\t\t<< final_df_1_fps >>\n')
#         print(final_df_1_fps)

#         print('\n\n\t\t<< final_df_5_fps >>\n')
#         print(final_df_5_fps)


#         base_save_path = '/raid/SSIM_RESULT/{}-SSIM_RESULT'.format(ssim_score_th)
#         ################ save df #################
#         df_save_path = os.path.join(base_save_path, target_anno.split('/')[-3], patient_full_name)
#         os.makedirs(df_save_path, exist_ok=True)
#         final_df_1_fps.to_csv(os.path.join(df_save_path, '{}-1FPS.csv'.format(patient_full_name)))
#         final_df_5_fps.to_csv(os.path.join(df_save_path, '{}-5FPS.csv'.format(patient_full_name)))

#         ############# video report ###############
#         report_per_video_save_path = os.path.join(base_save_path, target_anno.split('/')[-3], patient_full_name) # ssim_result/gangbuksamsung_127case/04_GS4_99_L_1
#         report_per_video(assets_df=final_df_1_fps, target_fps='1', time_th=time_th, ssim_score_th=ssim_score_th, save_path=report_per_video_save_path)
#         report_per_video(assets_df=final_df_5_fps, target_fps='5', time_th=time_th, ssim_score_th=ssim_score_th, save_path=report_per_video_save_path)

#         ############ patient report ##############
#         patient_name = '_'.join(patient[0].split('/')[-1].split('_')[:5])
#         report_per_patient_save_path = os.path.join(base_save_path, target_anno.split('/')[-3]) # ssim_result/gangbuksamsung_127case
#         report_per_patient(assets_df=final_df_1_fps, patient_name=patient_name, target_fps='1', time_th=time_th, ssim_score_th=ssim_score_th, save_path=report_per_patient_save_path, VIDEO_FPS=VIDEO_FPS)
#         report_per_patient(assets_df=final_df_5_fps, patient_name=patient_name, target_fps='5', time_th=time_th, ssim_score_th=ssim_score_th, save_path=report_per_patient_save_path, VIDEO_FPS=VIDEO_FPS)
        
#         ############## visualization #############
#         visual_save_path = os.path.join(base_save_path, target_anno.split('/')[-3], patient_full_name) # ssim_result/gangbuksamsung_127case/04_GS4_99_L_1
#         visual_sampling(final_df_1_fps, final_df_5_fps, window_size=5000, patient_no=patient_no, time_th=time_th, ssim_score_th=ssim_score_th, save_path=visual_save_path, VIDEO_FPS=VIDEO_FPS)
        
#         ################### gif ##################
#         gif_save_path = os.path.join(base_save_path, target_anno.split('/')[-3], patient_full_name, 'gif') # ssim_result/gangbuksamsung_127case/04_GS4_99_L_1/gif
#         gen_gif(assets_df=final_df_1_fps, target_fps=1, save_path=gif_save_path) 
#         gen_gif(assets_df=final_df_5_fps, target_fps=5, save_path=gif_save_path) 

#     ray.shutdown()


# if __name__ == '__main__':
#     import os
#     import glob
#     import natsort
#     import json

#     import numpy as np
#     import pandas as pd
#     from tqdm import tqdm
#     from itertools import groupby

#     import pickle
#     import matplotlib.pyplot as plt
#     from collections import defaultdict

#     import datetime

#     from visualization import visual_sampling
#     from report import report_per_video, report_per_patient
#     from gif import gen_gif

#     import sys

#     base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     sys.path.append(base_path)

#     print(base_path)
    
#     st = time.time()

#     # main(target_frame_base_path = '/raid/img_db/VIHUB/gangbuksamsung_127case', target_anno_base_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS', time_th=0, ssim_score_th=0.997)
#     # main(target_frame_base_path = '/raid/img_db/VIHUB/severance_1st', target_anno_base_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_1st/NRS', time_th=0, ssim_score_th=0.997)
#     main(target_frame_base_path = '/raid/img_db/VIHUB/severance_2nd', target_anno_base_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/severance_2nd/NRS', time_th=0, ssim_score_th=0.997)


#     ed = time.time()
#     elapsed_time = ed-st
#     print('{:.6f} seconds'.format(elapsed_time))
