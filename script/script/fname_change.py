import os
import shutil
from tqdm import tqdm

base_path = '/dataset/NRS'

for dset in os.listdir(base_path):
    dpath = base_path + f'/{dset}'
    
    for dtype in os.listdir(dpath):
        dpath2 = dpath + f'/{dtype}' + '/img'
        
        if os.path.exists(dpath2):
            patient_list = os.listdir(dpath2)
            
            if len(patient_list) > 0:
                for patient in patient_list:
                    dpath3 = dpath2 + f'/{patient}'
                    video_list = os.listdir(dpath3)
                    
                    if len(video_list) > 0:
                        for vname in video_list:
                            dpath4 = dpath3 + f'/{vname}'
                            file_list = os.listdir(dpath4)
                            
                            print(dpath4)
                            for fname in tqdm(file_list):
                                if len(fname) > 16:
                                    fname2 = fname.split('-')[-1]
                                    
                                    spath = dpath4 + f'/{fname}'
                                    tpath = dpath4 + f'/{fname2}'
                                    
                                    shutil.move(spath, tpath)
                                    
                            
        
        