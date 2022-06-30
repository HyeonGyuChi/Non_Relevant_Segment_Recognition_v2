

def main():
    import csv
    import cv2
    # import ray
    import time
    import imutils
    import datetime
    # from skimage.metrics import structural_similarity as ssim
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
    import json
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn


    from core.dataset import SubDataset
    from core.util.database import DBHelper
    from config.meta_db_config import subset_condition
    # from core.util.ssim import ssim_per_patient,cal_ssim_score
    import natsort

    import warnings
    from config.base_opts import parse_opts
    # warnings.filterwarnings("ignore")
    parser = parse_opts()
    args = parser.parse_args()

    from numba import jit,cuda

    # import os 
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    # print(device)


    json_data_list=[]
    patient_path=[]
    video_path=[]

    data_path = "/workspace/disk1/robot/vihub/img/"
    json_data_path = "/workspace/disk1/robot/vihub/anno/v1/"
    json_list = os.listdir(json_data_path)
    for i in range(len(os.listdir(json_data_path))):
        json_data_list.append(json_data_path + os.listdir(json_data_path)[i])
    

    for i in range(len(os.listdir(data_path))):
        patient_name = os.listdir(data_path)[i]
        if "meta" in patient_name:
            pass
        else: 
            patient_path.append( data_path + patient_name)
    # print("patient_path",patient_path)
    patient_path=natsort.natsorted(patient_path)
    # patient_path.remove("/workspace/disk1/robot/vihub/img/01_ViHUB_B1_R_1")
    # patient_path.remove("/workspace/disk1/robot/vihub/img/01_ViHUB_B1_R_2")
    # patient_path.remove("/workspace/disk1/robot/vihub/img/01_ViHUB_B1_R_42")
    # patient_path.remove("/workspace/disk1/robot/vihub/img/01_ViHUB_B1_R_52")
    # patient_path=["/workspace/disk1/robot/vihub/img/01_ViHUB_B1_R_64","/workspace/disk1/robot/vihub/img/01_ViHUB_B1_R_65","/workspace/disk1/robot/vihub/img/01_ViHUB_B1_R_66","/workspace/disk1/robot/vihub/img/01_ViHUB_B1_R_67"]
    print()
        
        
    for patient in patient_path: 
        print()
        print("patient",patient)
        video_path=[]
        for j in range(len(os.listdir(patient))):
            video_path.append(patient+"/"+os.listdir(patient)[j])
        # print("video_path",video_path)
        video_path=natsort.natsorted(video_path)

        for video in video_path:
            print("video",video)
            frame_path=[]
            for k in tqdm(os.listdir(video),desc="frame listing: "):
                frame_path.append(video+"/"+k)
                # print("frame_path length",len(frame_path))
            frame_path=natsort.natsorted(frame_path)
            frame_path_copy = frame_path.copy()


            # NRS - RS split
            for k in range(len(json_data_list)):
                 if json_data_list[k].startswith(json_data_path+video.split("/")[-1]):
                    ori_json = json_data_list[k]
                   
            # print("ori_json",ori_json)
            with open(ori_json, 'r') as f:
                nrs_frame_list=[]
                rs_frame_list=[]

                print("Total_frame",len(frame_path_copy))
                ori_json_data = json.load(f)
                total_json = ori_json_data['annotations']
                Total_anno = ori_json_data['totalFrame']
                print("Total_anno",Total_anno)
                idx=[]
                for l in range(len(total_json)):
                    idx.append(total_json[l]['start'])
                    idx.append(total_json[l]['end'])
                # print(idx)
                    
            for j in range(0,len(idx)//2,2):
                nrs_frame_list.append(frame_path_copy[idx[j]:idx[j+1]+1])
            nrs_frame_list = sum(nrs_frame_list, [])
            # print('nrs_frame_list',nrs_frame_list)
            print("nrs_frame_list",len(nrs_frame_list))

            rs_frame_list =  list(set(frame_path_copy) - set(nrs_frame_list))
            rs_frame_list = natsort.natsorted(rs_frame_list)
            print("rs_frame_list",len(rs_frame_list))

            print("sum: ", len(nrs_frame_list), "+", len(rs_frame_list), "=" , len(nrs_frame_list)+len(rs_frame_list))


            # # NRS
            print("=========START NRS SSIM=========")
            nrs_ssim_score_list = calculate_ssim_score(nrs_frame_list, 0, len(nrs_frame_list)-1)#.to(device)
            # if args.num_gpus > 1:
            #    nrs_ssim_score_list = torch.nn.DataParallel(calculate_ssim_score(nrs_frame_list, 0, len(nrs_frame_list)-1) , device_ids=list(range(args.num_gpus)))
            # print(nrs_ssim_score_list)
            NRS_duplicate_list=[]
            for k in range(len(nrs_ssim_score_list)):
                if nrs_ssim_score_list[k] > 0.997:
                    NRS_duplicate_list.append(nrs_ssim_score_list[k])
            NRS_duplicate = len(NRS_duplicate_list)
            NRS_non_duplicate = len(nrs_frame_list) - len(NRS_duplicate_list)

            # # RS
            print("=========START RS SSIM=========")
            rs_ssim_score_list = calculate_ssim_score(rs_frame_list, 0, len(rs_frame_list)-1)#.to(device)
            # if args.num_gpus > 1:
            #    rs_ssim_score_list = torch.nn.DataParallel(calculate_ssim_score(rs_frame_list, 0, len(rs_frame_list)-1), device_ids=list(range(args.num_gpus)))
            # print(rs_ssim_score_list)
            RS_duplicate_list=[]
            for k in range(len(rs_ssim_score_list)):
                if rs_ssim_score_list[k] > 0.997:
                    RS_duplicate_list.append(rs_ssim_score_list[k])
            RS_duplicate = len(RS_duplicate_list)
            RS_non_duplicate = len(rs_frame_list) - len(RS_duplicate_list)




            f = open('/workspace/disk1/meta/GPU_some.csv','a', newline='')
            wr = csv.writer(f)
            # patient	video	RS-non_duplicate	RS-duplicate	NRS-non_duplicate	NRS-duplicate
            wr.writerow([patient, video," ", RS_non_duplicate,RS_duplicate,NRS_non_duplicate,NRS_duplicate])
            # wr.writerow([patient.split("/")[-1], video.split("/")[-1]," ", Total_anno])
            f.close()


def calculate_ssim_score(target_ssim_list, st_idx, ed_idx):
    ssim_score_list = []
    import numpy as np
    import cupy as cp
    import cv2

    device = cp.cuda.Device(4).use()
    print(device)


    print("======calculate ssim score=====")
    for i in range(st_idx, ed_idx):
        prev_path, cur_path = target_ssim_list[i], target_ssim_list[i+1]
        print("prev_path",prev_path)
        print("cur_path ",cur_path)
        print()

        # Load the two input images
        imageA = cv2.imread(prev_path)
        imageB = cv2.imread(cur_path)

        # Convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)



        # Compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        # from skimage.metrics import structural_similarity as ssim
        # score, _ = ssim(grayA, grayB, full=True)
        # ssim_score_list.append(score)
        
        # import cucim.skimage.metrics.structural_similarity as cim
        from cucim.skimage.metrics import structural_similarity as cim
        grayA = cp.asarray(grayA)
        grayB = cp.asarray(grayB)
        score, _ = cim(grayA, grayB, full=True)
        ssim_score_list.append(score)


    return ssim_score_list




if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
    
    print("엥")
    main()

