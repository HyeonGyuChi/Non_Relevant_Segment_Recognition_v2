import os
base_path = "/workspace/disk1/robot/vihub/img/"
path_list = ["01_ViHUB_B1_R_60 / 01_ViHUB_B1_R_60_01"]
for i in range(len(path_list)):
    path = path_list[i]
    img_list = len(os.listdir(path))
    print("path",path,": ",img_list)