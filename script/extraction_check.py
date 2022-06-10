
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

    data_path = "/workspace/disk1/robot_vihub/img/"
    json_data_path = "/workspace/disk1/robot_vihub/anno/"
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
        
        
    for patient in patient_path: 
        print()
        print("patient",patient)
        video_path=[]
        for j in range(len(os.listdir(patient))):
            video_path.append(patient+"/"+os.listdir(patient)[j])
        # print("video_path",video_path)
        video_path=natsort.natsorted(video_path)

        for video in video_path:
            print("video  ",video)
            frame_path=[]
            for k in tqdm(os.listdir(video),desc="frame listing: "):
                frame_path.append(video+"/"+k)
                # print("frame_path length",len(frame_path))
            frame_path=natsort.natsorted(frame_path)
            print("Total_frame",len(frame_path))

            json_path = json_data_path + video.split("/")[-1] + "_NRS_30.json"
            # print("json  ",json_path)
            with open(json_path, 'r') as f:
                ori_json_data = json.load(f)
                Total_anno = ori_json_data['totalFrame']
                print("Total_anno ",Total_anno)
                        

            f = open('/workspace/disk1/meta//meta/length_info.csv','a', newline='')
            wr = csv.writer(f)
            # patient	video	RS-non_duplicate	RS-duplicate	NRS-non_duplicate	NRS-duplicate
            # wr.writerow([patient, video," ", Total_anno, RS_non_duplicate,RS_duplicate,NRS_non_duplicate,NRS_duplicate])
            wr.writerow([patient.split("/")[-1], video.split("/")[-1]," "," "," ", len(frame_path),Total_anno])
            f.close()


    
    ray.shutdown()


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
    
    main()

