import os
import json
import natsort
import pandas as pd
from core.util.parser import DBParser
from core.util.database import DBHelper
from config.meta_db_config import subset_condition

class Anno2Json():
    def __init__(self, args):
        self.args = args
        self.dp = DBParser(self.args, state='test')
        self.db_helper = DBHelper(args)

    def set_info(self,results,json_path):
        self.results = results
        self.json_path = json_path

    def make_json(self,version):
        if version == "autolabel":
            asset_df = self.db_helper.select(subset_condition["test"])
            for data in asset_df.values:
                # patient = data[2]
                patient = data[0]+"/"+data[1]+"/"+data[2]
                # patient_path = self.args.data_base_path + '/toyset/{}/{}/{}'.format(*data[:3])
                patient_path = self.args.data_base_path + '/{}/{}/{}'.format(*data[:3])
                for video_name in natsort.natsorted(os.listdir(patient_path)):
                    # new_data_video_value=(self.results.get(patient).get(video_name))
                    new_data_video_value=(self.results[patient][video_name])
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
                    
                    json_name = anno_base_path + "/"+ video_name + ".json"
                    with open(json_name, 'w') as f:
                        json.dump(new_json, f,indent="\t")
                            

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

            json_name = self.json_path
            with open(json_name, 'r') as f:
                json_data = json.load(f)
                total_json = json_data['annotations'] + new_data_anno
                total_json.sort(key = lambda item : item['start'])
                json_data['annotations']=total_json
            
            with open(json_name, 'w') as f:
                json.dump(json_data, f,indent="\t")
            

    def check_json_db_update(self,version):
        from config.base_opts import parse_opts
        parser = parse_opts()
        args = parser.parse_args()

        db_helper = DBHelper(args)
        BEFORE_df = db_helper.select(cond_info=None)
        # print(BEFORE_df)

        patient_list=[]
        for i in range(len(BEFORE_df["PATIENT"])):
            patient = BEFORE_df["SURGERY"][i]+"/"+BEFORE_df["SURGERY_TYPE"][i]+"/"+BEFORE_df["PATIENT"][i]
            patient_list.append(patient)

        
        # base_path =self.args.data_base_path + '/toyset/'
        base_path =self.args.data_base_path
        for i in range(len(patient_list)):
            patient = patient_list[i].split("/")[-1]
            patient_path = base_path + patient_list[i]
            for j in range(len(os.listdir(patient_path))):
                anno_path = patient_path + "/" +os.listdir(patient_path)[j] + "/anno"
                anno_list = os.listdir(anno_path)

                if "v1" == version:
                    json_path = os.listdir(anno_path + "/v1")
                    if json_path == None:
                        print("ssim labeling: PROBLEM")
                    else:
                        self.db_helper.update(
                            [['ANNOTATION_V1', 1],],
                            ["PATIENT==patient"],
                        )
                        
                elif "v2" == version:
                    pass

                elif "v3" == version:
                    json_path = os.listdir(anno_path + "/v3")
                    if json_path == None:
                        print("ssim labeling: PROBLEM")
                    else:
                        self.db_helper.update(
                            [['ANNOTATION_V3', 1],],
                            ["PATIENT==patient"],
                        )
                
        AFTER_df = self.db_helper.select(cond_info=None)
        print(AFTER_df)
       




# import os
# import json
# import natsort
# import pandas as pd
# from core.util.parser import DBParser
# from core.util.database import DBHelper
# from config.meta_db_config import subset_condition

# class Anno2Json():
#     def __init__(self, args):
#         self.args = args
#         self.dp = DBParser(self.args, state='test')
#         self.db_helper = DBHelper(args)

#     def set_info(self,results,json_path):
#         self.results = results
#         self.json_path = json_path

#     def make_json(self,version):
#         if version == "autolabel":
#             asset_df = self.db_helper.select(subset_condition["test"])
#             for data in asset_df.values:
#                 patient = data[2]
#                 patient_path = self.args.data_base_path + '/toyset/{}/{}/{}'.format(*data[:3])
#                 for video_name in natsort.natsorted(os.listdir(patient_path)):
#                     # print("patient",patient)
#                     # print("video_name",video_name)

#                     # new_data_video_value=(self.results.get(patient).get(video_name))
#                     new_data_video_value= self.results[patient][video_name]
#                     # print("new_data_video_value",new_data_video_value)

#                     new_data_totalframe=len(new_data_video_value[0])
#                     print("new_data_totalframe",new_data_totalframe)

#                     new_data_label=new_data_video_value[1]
#                     new_data_anno=[]
#                     idx_list_nrs=[]
#                     for k in range(len(new_data_label)):
#                         if new_data_label[k]==1:
#                             idx_list_nrs.append(k)
#                     idx_list_nrs_copy = idx_list_nrs.copy()

#                     for k in range(len(idx_list_nrs)-2):
#                         if idx_list_nrs[k]+1==idx_list_nrs[k+1]:
#                             if idx_list_nrs[k+1]+1 < idx_list_nrs[k+2]:
#                                 pass
#                             else:
#                                 idx_list_nrs_copy.remove(idx_list_nrs[k+1])
#                     for k in range(0,len(idx_list_nrs_copy)-1,2):
#                         new_data_start_end = {"start": idx_list_nrs_copy[k],"end": idx_list_nrs_copy[k+1], "code":1}
#                         new_data_anno.append(new_data_start_end)

#                     new_json = {
#                     "totalFrame": new_data_totalframe,
#                     "frameRate": 30,
#                     "width": 1920,
#                     "height": 1080,
#                     "_id": "61afe4e12bd5d3001b3578dc",
#                     "annotations": new_data_anno,
#                     "annotationType": "NRS",
#                     "createdAt":"",  
#                     "updatedAt": "", 
#                     "annotator": "30",
#                     "name": video_name,
#                     "label": {"1": "NonRelevantSurgery"}
#                     }

#                     anno_base_path = patient_path + f'/{video_name}/anno/v1'
#                     if os.path.exists(anno_base_path):
#                         pass
#                     else:
#                         os.mkdir(anno_base_path)
                    
#                     if "R_" in anno_base_path:
#                         json_name = anno_base_path + "/"+ video_name + "_TBE_30.json"
#                         print("json_name", json_name)                     
#                         with open(json_name, 'w') as f:
#                             json.dump(new_json, f,indent="\t")
#                     elif "L_" in anno_base_path:
#                         json_name = anno_base_path + "/"+ video_name + "_NRS_30.json"
#                         print("json_name", json_name)
#                         with open(json_name, 'w') as f:
#                             json.dump(new_json, f,indent="\t")

#         elif version == "ssim":
#             # NRS 이미지만 ssim 계산 후 annotation
#             index_list = []
#             for data in self.results.values:
#                 ssim = data[5]
#                 index = data[0]
#                 if ssim > 0.997:
#                     index_list.append(index)

#             index_list_copy = index_list.copy()
#             new_data_anno=[]
#             for i in range(len(index_list)-2):
#                 if index_list[i]+1==index_list[i+1]:
#                     if index_list[i+1]+1 < index_list[i+2]:
#                         pass
#                     else:
#                         index_list_copy.remove(index_list[i+1])
#             for i in range(0,len(index_list_copy)-1,2):
#                 new_data_start_end = {"start": index_list_copy[i],"end": index_list_copy[i+1], "code":2}
#                 new_data_anno.append(new_data_start_end)

#             json_name = self.json_path
#             with open(json_name, 'r') as f:
#                 json_data = json.load(f)
#                 total_json = json_data['annotations'] + new_data_anno
#                 total_json.sort(key = lambda item : item['start'])
#                 json_data['annotations']=total_json
            
#             with open(json_name, 'w') as f:
#                 json.dump(json_data, f,indent="\t")
            

#     def check_json_db_update(self,version):
#         from config.base_opts import parse_opts
#         parser = parse_opts()
#         args = parser.parse_args()

#         db_helper = DBHelper(args)
#         BEFORE_df = db_helper.select(cond_info=None)
#         # print(BEFORE_df)

#         patient_list=[]
#         for i in range(len(BEFORE_df["PATIENT"])):
#             patient = BEFORE_df["SURGERY"][i]+"/"+BEFORE_df["SURGERY_TYPE"][i]+"/"+BEFORE_df["PATIENT"][i]
#             patient_list.append(patient)


#         base_path = "../core/dataset/NRS/toyset/"
#         for i in range(len(patient_list)):
#             patient = patient_list[i].split("/")[-1]
#             patient_path = base_path + patient_list[i]
#             for j in range(len(os.listdir(patient_path))):
#                 anno_path = patient_path + "/" +os.listdir(patient_path)[j] + "/anno"
#                 anno_list = os.listdir(anno_path)

#                 if "v1" == version:
#                     json_path = os.listdir(anno_path + "/v1")
#                     if json_path == None:
#                         print("ssim labeling: PROBLEM")
#                     else:
#                         self.db_helper.update(
#                             [['ANNOTATION_V1', 1],],
#                             ["PATIENT==patient"],
#                         )
#                         AFTER_df = self.db_helper.select(cond_info=None)
#                         print(AFTER_df)

#                 elif "v2" == version:
#                     pass

#                 elif "v3" == version:
#                     json_path = os.listdir(anno_path + "/v3")
#                     if json_path == None:
#                         print("ssim labeling: PROBLEM")
#                     else:
#                         self.db_helper.update(
#                             [['ANNOTATION_V3', 1],],
#                             ["PATIENT==patient"],
#                         )
#                         AFTER_df = self.db_helper.select(cond_info=None)
#                         print(AFTER_df)
       







