import os
from glob import glob


def get_inference_model_path(restore_path, load_type='best'):
    model_path = ''
    
    print('restore_path : ', restore_path)
    
    ckpoint_path = os.path.join(restore_path + '/checkpoints', '*.pth')
    ckpts = glob(ckpoint_path)
    # print('ckpoint_path_list : ', ckpts)

    for f_name in ckpts :
        if f_name.find('best') != -1 :
            model_path = f_name
            
    print('model_path : ', model_path)
            
    return model_path 