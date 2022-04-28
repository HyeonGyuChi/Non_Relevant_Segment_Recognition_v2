import cv2
import ray
import time
import imutils
import datetime
import os
import json
import glob
import pandas as pd
import natsort
from skimage.metrics import structural_similarity as ssim
# from core.util.parser import AssetParser

def ssim_per_patient(full_patient_list):
    import warnings
    warnings.filterwarnings("ignore")

    from config.base_opts import parse_opts
    parser = parse_opts()
    args = parser.parse_args()

    full_patient_list = natsort.natsorted(full_patient_list)
    for patient in full_patient_list: 
        print("patient",patient)
        per_patient_list = []

        video_list = os.listdir(patient)
        video_path=[]
        for i in range(len(video_list)):
            video_path.append(patient+"/"+video_list[i])
        print("video_path",video_path)
        
        for video in video_path:
            # NRS 이미지만 ssim 계산
            
            ori_anno = video+"/anno/v1/"+ os.listdir(video+"/anno/v1")[0]
            print('\n+[origin anno]  : {}'.format(ori_anno))

            target_anno_path = video+"/anno/v3/"
            if os.path.exists(target_anno_path):
                target_anno = target_anno_path + os.listdir(video+"/anno/v1")[0]
            else:
                os.mkdir(target_anno_path)
                target_anno = target_anno_path + os.listdir(video+"/anno/v1")[0]
                import shutil
                shutil.copy(ori_anno,target_anno)
            print('+[target anno]  : {}'.format(target_anno))

            print('+[target frames] : {}'.format(video))
            target_frames_list = os.listdir(video+"/img")
            target_frames=[]
            for j in range(len(target_frames_list)):
                target_frames.append(video+"/img/" +target_frames_list[j])
            target_frames = natsort.natsorted(target_frames)

            gt_list, ssim_list = get_anno_list(target_anno=target_anno, total_len=len(target_frames), time_th=0)    

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
            time_idx = [idx_to_time(idx, 30) for idx in frame_idx]
            
            patient_df['frame_idx'] = frame_idx
            patient_df['time_idx'] = time_idx
            patient_df = patient_df[['frame_idx', 'time_idx', 'frame_path', 'gt', 'ssim']]
            print('\n\n\t\t<< patient df >>\n')
            print(patient_df)

            patient_df_copy =  pd.DataFrame()
            with open(target_anno, 'r') as f:
                json_data = json.load(f)
                for i in range(len(json_data['annotations'])):
                    start = patient_df['frame_idx'] >= json_data['annotations'][i]["start"]
                    end = patient_df['frame_idx'] <= json_data['annotations'][i]["end"]
                    subset_df = patient_df[start & end]              
                    patient_df_copy=pd.concat([patient_df_copy, subset_df], axis = 0)
            # print('\n\n\t\t<< patient df_NRS >>\n')
            # print(patient_df_copy)

            # ######### calculate ssim score #########
            final_df_1_fps = compute_ssim(target_assets_df=patient_df_copy, ssim_score_th=0.997, n_cpu=10)
            print('\n\n\t\t<< final_df_1_fps >>\n')
            print(final_df_1_fps) 

            # final_df_5_fps = SSIM.compute_ssim(target_assets_df=patient_df, ssim_score_th=0.997, n_cpu=50)
            # print('\n\n\t\t<< final_df_5_fps >>\n')
            # print("final_df_1_fps",final_df_5_fps) 
            
            print('\n\n\t\t<< JSON 수정 >>')
            print(target_anno)
            from core.util.anno2json import Anno2Json
            annotation_to_json = Anno2Json(args,final_df_1_fps,target_anno)
            annotation_to_json.make_json(version="ssim")

    annotation_to_json.check_json_db_update(version="v3")
    print()
    print()
    print()

        




   
@ray.remote
def cal_ssim_score(target_ssim_list, st_idx, ed_idx):
    ssim_score_list = []
    import numpy as np
    import cv2

    print("======calculate ssim score=====")
    for i in range(st_idx, ed_idx):
        prev_path, cur_path = target_ssim_list[i], target_ssim_list[i+1]

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


def compute_ssim(target_assets_df, ssim_score_th, n_cpu):
    import numpy as np
    import pandas as pd
    print("======compute ssim score=====")

    # ray :)
    target_ssim_df = target_assets_df[target_assets_df.ssim == 1][['frame_path']]
    target_ssim_list = target_ssim_df['frame_path'].values.tolist()

    split = len(target_ssim_list) // n_cpu
    st_list = [0 + split*i for i in range(n_cpu)]
    ed_list = [split*(i+1) for i in range(n_cpu)]
    ed_list[-1] = len(target_ssim_list)-1

    ray_target_ssim_list = ray.put(target_ssim_list)
    results = ray.get([cal_ssim_score.remote(ray_target_ssim_list, st_list[i], ed_list[i]) for i in range(n_cpu)])

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


def idx_to_time(idx, fps):
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




