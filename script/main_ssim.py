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


def main():
    from core.dataset import SubDataset
    from core.util.database import DBHelper
    from config.meta_db_config import subset_condition
    from core.util.ssim import ssim_per_patient
    import warnings
    from config.base_opts import parse_opts
    warnings.filterwarnings("ignore")
    parser = parse_opts()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    ray.init(num_cpus=60)

    # sub_condition: 'ssim': ['ANNOTATION_V3==0']
    ssimset = SubDataset(args, state='ssim', sample_type=args.sample_type)

    # get_patient_list
    full_patient_list = ssimset.get_patient_list()

    # SSIM per patient + make json + RESULT CHECK and DB UPDATE
    ssim_per_patient(full_patient_list)

    ray.shutdown()


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
    
    main()

