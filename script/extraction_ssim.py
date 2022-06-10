
def main():
    import csv
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
    import json
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader


    from core.dataset import SubDataset
    from core.util.database import DBHelper
    from config.meta_db_config import subset_condition
    from core.util.ssim import ssim_per_patient,cal_ssim_score
    import natsort

    import warnings
    from config.base_opts import parse_opts
    warnings.filterwarnings("ignore")
    parser = parse_opts()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    ray.init(num_cpus=60)
    num_cpus=60


    json_data_list=[]
    patient_path=[]
    video_path=[]
    frame_path=[]

    data_path = "/workspace/disk1/robot/vihub/img/"
    json_data_path = "/workspace/disk1/robot/vihub/anno/"
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
            for k in tqdm(os.listdir(video),desc="frame listing: "):
                frame_path.append(video+"/"+k)
                # print("frame_path length",len(frame_path))
            frame_path=natsort.natsorted(frame_path)
            frame_path_copy = frame_path.copy()


            # NRS - RS split
            for k in range(len(json_data_list)):
                if video.split("/")[-1] in json_data_list[k]:
                    ori_json = json_data_list[k]
                    print("json",ori_json)

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
                        
            for k in range(0,len(idx)//2,2):
                nrs_frame_list.append(frame_path_copy[idx[k]:idx[k+1]+1])
            nrs_frame_list = sum(nrs_frame_list, [])
            # print('nrs_frame_list',nrs_frame_list)
            print("nrs_frame_list",len(nrs_frame_list))

            rs_frame_list =  list(set(frame_path_copy) - set(nrs_frame_list))
            print("rs_frame_list",len(rs_frame_list))

            print("sum: ", len(nrs_frame_list), "+", len(rs_frame_list), "=" , len(nrs_frame_list)+len(rs_frame_list))


            # # NRS
            print("=========START NRS SSIM=========")
            ray_target_ssim_list = ray.put(nrs_frame_list)
            nrs_ssim_score_list = ray.get([cal_ssim_score.remote(ray_target_ssim_list, 0, len(nrs_frame_list)-1) for i in range (num_cpus)])
            # print(nrs_ssim_score_list)
            NRS_duplicate_list=[]
            for k in range(len(nrs_ssim_score_list)):
                if nrs_ssim_score_list[0][k] > 0.997:
                    NRS_duplicate_list.append(nrs_ssim_score_list[0][k])
            NRS_duplicate = len(NRS_duplicate_list)
            NRS_non_duplicate = len(nrs_frame_list) - len(NRS_duplicate_list)


            # # RS
            print("=========START RS SSIM=========")
            ray_target_ssim_list = ray.put(rs_frame_list)
            rs_ssim_score_list = ray.get([cal_ssim_score.remote(ray_target_ssim_list, 0, len(rs_frame_list)-1) for i in range (num_cpus)])
            # print(rs_ssim_score_list)
            RS_duplicate_list=[]
            for k in range(len(rs_ssim_score_list)):
                if rs_ssim_score_list[0][k] > 0.997:
                    RS_duplicate_list.append(rs_ssim_score_list[0][k])
            RS_duplicate = len(RS_duplicate_list)
            RS_non_duplicate = len(rs_frame_list) - len(RS_duplicate_list)


            f = open('/workspace/disk1/meta//meta/nrs_info_0609.csv','a', newline='')
            wr = csv.writer(f)
            # patient	video	RS-non_duplicate	RS-duplicate	NRS-non_duplicate	NRS-duplicate
            wr.writerow([patient, video," ", RS_non_duplicate,RS_duplicate,NRS_non_duplicate,NRS_duplicate])
            # wr.writerow([patient.split("/")[-1], video.split("/")[-1]," ", Total_anno])
            f.close()


    
    ray.shutdown()


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
    
    main()

