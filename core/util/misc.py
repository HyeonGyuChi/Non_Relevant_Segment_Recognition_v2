import os
from glob import glob
import natsort


def get_inference_model_path(restore_path, load_type='best'):
    model_path = ''
    
    print('restore_path : ', restore_path)
    
    ckpoint_path = os.path.join(restore_path + '/checkpoints', '*.pth')
    ckpts = glob(ckpoint_path)
    ckpts = natsort.natsorted(ckpts)
    print('ckpoint_path_list : ', ckpts)

    for f_name in ckpts :
        if f_name.find('best') != -1 :
            model_path = f_name
            
    print('model_path : ', model_path)
            
    return model_path 

def save_dict_to_csv(results_dict, save_path):
    import os
    import csv
    import json
    import pandas as pd
    from core.util.parser.parser import FileLoader # file load helper

    results_df = pd.DataFrame.from_dict([results_dict]) # dict to df
    results_df = results_df.reset_index(drop=True)

    merged_df = results_df
    if os.path.isfile(save_path): # append
        f_loader = FileLoader()
        f_loader.set_file_path(save_path)
        saved_df = f_loader.load()

        saved_df.drop(['Unnamed: 0'], axis = 1, inplace = True) # to remove Unmaned : 0 colume

        merged_df = pd.concat([saved_df, results_df], ignore_index=True, sort=False)
        
        merged_df.to_csv(save_path, mode='w')

        # print(merged_df)

    merged_df.to_csv(save_path, mode='w')
