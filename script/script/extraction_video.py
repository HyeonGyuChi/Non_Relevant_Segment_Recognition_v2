# ######### Robot VIHUB Dataset(86case) Video -> Frame (ffmpeg) #########
import os
import csv
import ffmpeg
from pprint import pprint



# # Base path
# data_path = "/workspace/mnt/jwlee/"

# save_path = "/workspace/disk2/extraction_16case/" 
# robot_vihub_path_1 = data_path + "usb1/IDC_0520"
# robot_vihub_path_all= [robot_vihub_path_1]

# # save_path = "/workspace/disk1/extraction_70case/" 
# # robot_vihub_path_2 = data_path + "usb2/IDC_0520/ViHUB"
# # robot_vihub_path_all= [robot_vihub_path_2]

# # robot_vihub_path_all= [robot_vihub_path_1,robot_vihub_path_2]



# for m in range(len(robot_vihub_path_all)):
#     folder = robot_vihub_path_all[m]

#     # Patient path
#     patient_list = os.listdir(folder)
#     patient_list_path=[]
#     for i in range(len(patient_list)): 
#         patient_list_path.append(folder + "/" + patient_list[i] )

#     for i in range(len(patient_list_path)):
#         print("\nPATIENT: ", patient_list_path[i])
#         video_path_list=[]
#         video_list = os.listdir(patient_list_path[i])
#         for j in range(len(video_list)):
#             video_path_list.append(patient_list_path[i] + "/" + video_list[j])
#         # print(video_path_list)

#         # Extraction 
#         for j in range(len(video_path_list)):
#             print("VIDEO:  ", video_path_list[j])
#             save_frame_path = save_path + patient_list_path[i].split("/")[-1] + "/" + video_path_list[j].split("/")[-1].split(".")[0]
#             save_metta_path = save_path + "meta/"
#             if not os.path.exists(save_metta_path):
#                 os.makedirs(save_metta_path)
#             if not os.path.exists(save_frame_path):
#                 os.makedirs(save_frame_path)

#             # Extraction Frame
#             # 치현님
#             # ffmpeg -i "$fname" -filter:v scale='ih*dar:ih' -threads 2 "frames/${fname%.*}"/frame%010d.jpg
            
#             # ffmpeg Helper
#             # 'cut_frame_total': ['ffmpeg', '-i', self.video_path, '-start_number', '0', '-vf', 'scale=512:512', self.results_dir + '/{}-%010d.jpg'.format(args.get('save_name', -100))],
#             # 'cut_frame_total': ['ffmpeg', '-i', self.video_path, '-start_number', '0', '-vsync', '0', '-vf', 'scale=512:512', self.results_dir + '/{}-%010d.jpg'.format(args.get('save_name', -100))],
#             # 'cut_frame_1fps': ['ffmpeg', '-i', self.video_path, '-s', '224x224', '-vf', 'fps=1', self.results_dir + '/frame-%010d.jpg'],
#             # 'extract_frame_by_index': ['ffmpeg', '-i', self.video_path, '-vf', '"select=eq(n\,{})"'.format(args.get('frame_index', self.EXCEPTION_NUM)), '-vframes', '1', self.results_dir + '/extracted_frame-{}.jpg'.format(args.get('frame_index', self.EXCEPTION_NUM))],
#             # 'extract_frame_by_time': ['ffmpeg', '-i', self.video_path, '-ss', '{}'.format(args.get('time', self.EXCEPTION_NUM)), '-frames:v', '1', self.results_dir + '/extracted_frame-{}.jpg'.format(args.get('time', self.EXCEPTION_NUM))],
#             cmd_str_frame = "ffmpeg -i " + video_path_list[j] + " " + "-filter:v scale='512:512' -threads 2 " + save_frame_path + "/{}-%010d.jpg".format(video_path_list[j].split("/")[-1].split(".")[0])
#             os.system(cmd_str_frame)
            


#             # Extraction Meta Info
#             # cmd_str_Meta= "ffprobe -v error -select_streams 0:8 -print_format json -show_streams /workspace/mnt/jwlee/usb2/IDC_0520/ViHUB/01_ViHUB_B1_R_2/01_ViHUB_B1_R_2_03.mp4"
#             # os.system(cmd_str_Meta)

#             Meta_json = ffmpeg.probe(video_path_list[j])['streams'][0]
#             fps = int(Meta_json['avg_frame_rate'].split("/")[0]) / int(Meta_json['avg_frame_rate'].split("/")[1])
#             total_frame = Meta_json['nb_frames']
 
#             f = open(save_metta_path + 'meta_info.csv','a', newline='')
#             wr = csv.writer(f)
#             wr.writerow([j+1,patient_list_path[i].split("/")[-1], video_path_list[j].split("/")[-1].split(".")[0]," "," ",fps,total_frame])
#             f.close()







# 추가 비디오 
# video_path_list = ["/workspace/disk1/22.05.30/22.05.30/Video/01_ViHUB_B1_R_49_03.mp4","/workspace/disk1/22.05.30/22.05.30/Video/01_ViHUB_B1_R_49_04.mp4"]
video_path_list = ["/workspace/mnt/jwlee/usb1/IDC_0520/01_ViHUB_B6_R_133/01_ViHUB_B6_R_133_01.mp4"]
save_path = "/workspace/disk1/01_ViHUB_B6_R_133_01_vframes_number/"


# Extraction 
for j in range(len(video_path_list)):
    print("VIDEO:  ", video_path_list[j])
    save_frame_path = save_path + video_path_list[j].split("/")[-1].split(".")[0]
    save_metta_path = save_path + "meta/"
    if not os.path.exists(save_metta_path):
        os.makedirs(save_metta_path)
    if not os.path.exists(save_frame_path):
        os.makedirs(save_frame_path)

    # Extraction Meta Info
    Meta_json = ffmpeg.probe(video_path_list[j])['streams'][0]
    # print("Meta_json",Meta_json)
    fps = int(Meta_json['avg_frame_rate'].split("/")[0]) / int(Meta_json['avg_frame_rate'].split("/")[1])
    total_frame = Meta_json['nb_frames']
    print("fps",fps)
    print("total_frame",total_frame)
    
    # Extraction Frame
    cmd_str_frame = "ffmpeg -i " + video_path_list[j] + " " +"-vframes " + str(total_frame) +" "+"-filter:v scale='512:512' -threads 2 " + save_frame_path + "/{}-%010d.jpg".format(video_path_list[j].split("/")[-1].split(".")[0])
    os.system(cmd_str_frame)


    f = open(save_metta_path + '01_ViHUB_B6_R_133_01_meta_info.csv','a', newline='')
    wr = csv.writer(f)
    wr.writerow([j+1,video_path_list[j].split("/")[-1].split(".")[0][:-3], video_path_list[j].split("/")[-1].split(".")[0]," "," ",fps,total_frame])
    f.close()