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


def main(version,surgery,db):
    from core.dataset import SubDataset
    from core.dataset import RobotDataset
    from core.util.database import DBHelper
    from config.meta_db_config import subset_condition
    from core.util.ssim import ssim_per_patient
    import warnings
    from config.base_opts import parse_opts
    warnings.filterwarnings("ignore")
    parser = parse_opts()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list


    if version == "cpu":
        ray.init(num_cpus=60)

        if db == "yes":
            pass
            # # # DB로 할때
            # ssimset = SubDataset(args, state='ssim', sample_type=args.sample_type)
            # full_patient_list = ssimset.get_patient_list()
            # ssim_per_patient(full_patient_list, db="yes")


        elif db == "no":
            data_path =  args.data_base_path + "/" + surgery + "/vihub/img/" 
            patient_list = os.listdir(data_path)
            patient_path =[]
            for i in range(len(patient_list)):
                patient_path.append(data_path+patient_list[i])
            patient_path = natsort.natsorted(patient_path)
            ssim_per_patient(patient_path,version,surgery,db)

        ray.shutdown()


    elif version == "gpu":

        if db == "yes":
            pass
            # # # DB로 할때
            # ssimset = SubDataset(args, state='ssim', sample_type=args.sample_type)
            # full_patient_list = ssimset.get_patient_list()
            # ssim_per_patient(full_patient_list, db="yes")


        elif db == "no":
            data_path =  args.data_base_path + "/" + surgery + "/vihub/img/" 
            patient_list = os.listdir(data_path)
            patient_path =[]
            for i in range(len(patient_list)):
                patient_path.append(data_path+patient_list[i])
            patient_path = natsort.natsorted(patient_path)
            ssim_per_patient(patient_path,version,surgery,db)



if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
    
    # main(version="cpu")
    main(version="cpu", surgery = "robot",db="no")

