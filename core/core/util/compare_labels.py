import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



def compare_labels(patient):
    import os
    import sys
    sys.path.append((os.path.dirname(__file__)))
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


    from core.util.visualization import VisualTool
    from config.base_opts import parse_opts
    parser = parse_opts()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list


    # base_path = args.data_base_path + "/toyset/"
    base_path = args.data_base_path 
    video_list = os.listdir(base_path + patient) 
    
    for i in range(len(video_list)):
        # json check
        # 01_VIHUB1.2_A9_L_24_02_NRS_31.json 

        try: 
            gt_json_list = os.listdir(base_path + patient +"/"+video_list[i] + "/anno/org/")
            for j in range(len(gt_json_list)):
                if video_list[i] in gt_json_list[j]:
                    gt_json = base_path + patient + "/"+video_list[i] +"/anno/org/" + gt_json_list[j]
                    with open(gt_json, 'r') as gt:
                        gt_json = json.load(gt)
                        # gt_json_anno = gt_json['annotations']


            auto_json_list = os.listdir(base_path + patient +"/"+video_list[i] + "/anno/v1/")
            for j in range(len(auto_json_list)):
                if video_list[i] in auto_json_list[j]:
                    auto_json = base_path + patient + "/"+video_list[i] +"/anno/v1/" + auto_json_list[j]
                    with open(auto_json, 'r') as auto:
                        auto_json = json.load(auto)
                        # auto_json_anno = auto_json['annotations']

    
            ssim_json_list = os.listdir(base_path + patient + "/"+video_list[i] +"/anno/v3/")
            for j in range(len(ssim_json_list)):
                if video_list[i] in ssim_json_list[j]:
                    ssim_json = base_path + patient + "/"+video_list[i] +"/anno/v3/" + ssim_json_list[j]
                    with open(ssim_json, 'r') as ssim:
                        ssim_json = json.load(ssim)
                        # ssim_json_anno = ssim_json['annotations']

            gt_list = json2list(gt_json)
            auto_list = json2list(auto_json)
            ssim_list = json2list(ssim_json)


        except:
            auto_json_list = os.listdir(base_path + patient +"/"+video_list[i] + "/anno/v1/")
            for j in range(len(auto_json_list)):
                if video_list[i] in auto_json_list[j]:
                    auto_json = base_path + patient + "/"+video_list[i] +"/anno/v1/" + auto_json_list[j]
                    with open(auto_json, 'r') as auto:
                        auto_json = json.load(auto)
                        # auto_json_anno = auto_json['annotations']

    
            ssim_json_list = os.listdir(base_path + patient + "/"+video_list[i] +"/anno/v3/")
            for j in range(len(ssim_json_list)):
                if video_list[i] in ssim_json_list[j]:
                    ssim_json = base_path + patient + "/"+video_list[i] +"/anno/v3/" + ssim_json_list[j]
                    with open(ssim_json, 'r') as ssim:
                        ssim_json = json.load(ssim)
                        # ssim_json_anno = ssim_json['annotations']

            auto_list = json2list(auto_json)
            ssim_list = json2list(ssim_json)
            gt_list = [0 for i in range(len(auto_list))]

        # frame slice error  해결
        # gt 길이 기준
        gt_list_len = len(gt_list)
        auto_list_len = len(auto_list)
        if gt_list_len != auto_list_len:
            gap = gt_list_len - auto_list_len
            for k in range (gap):
                auto_list.append(0)
                ssim_list.append(0)

        patient_labels = {patient+"/"+video_list[i]: {"gt_list": gt_list,"auto_list": auto_list,"ssim_list": ssim_list, }}

        #plot
        # save_root_path = "/workspace/jiwoo/NRS/core/util/logs/toyset_inference/"
        save_root_path = "/workspace/jiwoo/NRS/core/util/logs/subdataset_inference/"
        save_path_folder = '{}/{}'.format(save_root_path, patient)
        if  os.path.exists(save_path_folder):
            pass
        else:
            surgery = patient.split("/")[0]
            surgen = patient.split("/")[1]
            patient_name = patient.split("/")[2]
            if not os.path.exists(save_root_path):
                os.mkdir(save_root_path)
            if not os.path.exists(save_root_path+surgery):
                os.mkdir(save_root_path+surgery)
            if not os.path.exists(save_root_path+surgery+ "/"+surgen):
                os.mkdir(save_root_path+surgery+ "/"+surgen)
            if not os.path.exists(save_root_path+surgery+ "/"+surgen+patient_name):
                os.mkdir(save_root_path+surgery+ "/"+surgen+"/"+patient_name)


        save_path = '{}/{}/{}.png'.format(save_root_path, patient, video_list[i])
        visual_tool = VisualTool(args)
        visual_tool.visualize_compare(patient_labels,save_path)




def json2list(anno):
    # print("json2list anno", anno)
    totalFrame = anno['totalFrame']
    annotations = anno['annotations']

    label_list = [0 for i in range(totalFrame)]
    for i in range(len(annotations)):
        if annotations[i]['code'] == 1:
            #NRS
            start_idx = annotations[i]['start']
            end_idx = annotations[i]['end']
            for j in range(start_idx, end_idx):
                label_list[j] = 1

        elif annotations[i]['code'] == 2:
            #ssim
            start_idx = annotations[i]['start']
            end_idx = annotations[i]['end']
            for j in range(start_idx, end_idx):
                label_list[j] = 2

    return label_list

