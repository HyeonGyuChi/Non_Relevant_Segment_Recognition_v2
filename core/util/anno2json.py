import os
import json
import natsort
import pandas as pd
from core.util.parser import DBParser
from core.util.database import DBHelper
from config.meta_db_config import subset_condition

class Anno2Json():
    def __init__(self, args,results,target_anno):
        self.args = args
        self.results = results
        self.target_anno = target_anno
        self.dp = DBParser(self.args, state='test')
        self.db_helper = DBHelper(args)

    def make_json(self,version):
        if version == "autolabel":
            asset_df = self.db_helper.select(subset_condition["test"])
            for data in asset_df.values:
                patient = data[2]
                patient_path = self.args.data_base_path + '/toyset/{}/{}/{}'.format(*data[:3])
                for video_name in natsort.natsorted(os.listdir(patient_path)):
                    new_data_video_value=(self.results.get(patient).get(video_name))
                    new_data_totalframe=len(new_data_video_value[0])
                    new_data_label=new_data_video_value[1]
                    new_data_anno=[]
                    idx_list_nrs=[]
                    for k in range(len(new_data_label)):
                        if new_data_label[k]==1:
                            idx_list_nrs.append(k)
                    idx_list_nrs_copy = idx_list_nrs.copy()

                    for k in range(len(idx_list_nrs)-2):
                        if idx_list_nrs[k]+1==idx_list_nrs[k+1]:
                            if idx_list_nrs[k+1]+1 < idx_list_nrs[k+2]:
                                pass
                            else:
                                idx_list_nrs_copy.remove(idx_list_nrs[k+1])
                    for k in range(0,len(idx_list_nrs_copy)-1,2):
                        new_data_start_end = {"start": idx_list_nrs_copy[k],"end": idx_list_nrs_copy[k+1], "code":1}
                        new_data_anno.append(new_data_start_end)

                    new_json = {
                    "totalFrame": new_data_totalframe,
                    "frameRate": 30,
                    "width": 1920,
                    "height": 1080,
                    "_id": "61afe4e12bd5d3001b3578dc",
                    "annotations": new_data_anno,
                    "annotationType": "NRS",
                    "createdAt":"",  
                    "updatedAt": "", 
                    "annotator": "30",
                    "name": video_name,
                    "label": {"1": "NonRelevantSurgery"}
                    }

                    anno_base_path = patient_path + f'/{video_name}/anno/v1'
                    if os.path.exists(anno_base_path):
                        pass
                    else:
                        os.mkdir(anno_base_path)
                    
                    if "R_" in anno_base_path:
                        json_name = anno_base_path + "/"+ video_name + "_TBE_30.json"
                        print("json_name", json_name)                     
                        with open(json_name, 'w') as f:
                            json.dump(new_json, f)
                    elif "L_" in anno_base_path:
                        json_name = anno_base_path + "/"+ video_name + "_NRS_30.json"
                        print("json_name", json_name)
                        with open(json_name, 'w') as f:
                            json.dump(new_json, f)

        elif version == "ssim":
            # NRS 이미지만 ssim 계산 후 annotation
            index_list = []
            for data in self.results.values:
                ssim = data[5]
                index = data[0]
                if ssim > 0.997:
                    index_list.append(index)

            index_list_copy = index_list.copy()
            new_data_anno=[]
            for i in range(len(index_list)-2):
                if index_list[i]+1==index_list[i+1]:
                    if index_list[i+1]+1 < index_list[i+2]:
                        pass
                    else:
                        index_list_copy.remove(index_list[i+1])
            for i in range(0,len(index_list_copy)-1,2):
                new_data_start_end = {"start": index_list_copy[i],"end": index_list_copy[i+1], "code":2}
                new_data_anno.append(new_data_start_end)

            json_path = self.target_anno
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                total_json = json_data['annotations'] + new_data_anno
                total_json.sort(key = lambda item : item['start'])
                json_data['annotations']=total_json
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f,indent="\t")





