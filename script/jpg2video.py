
# #jpg2video
pathIn=[]

# for i in range(0,197):
#     pathIn.append('/workspace/disk1/robot_vihub/img/01_ViHUB_B6_R_133/01_ViHUB_B6_R_133_01/01_ViHUB_B6_R_133_01-00000' + str(i)+ ".jpg")
#     # 000 000 000 1

import os
path = '/workspace/disk1/robot_vihub/img/01_ViHUB_B6_R_133/01_ViHUB_B6_R_133_01/'
file_list=os.listdir(path)
for i in range(len(file_list)):
    pathIn.append(path+file_list[i])
pathIn = pathIn[0:197]
pathOut = '/workspace/disk1/01_ViHUB_B6_R_133_01_gan.mp4'
fps = 30

import cv2
frame_array = []
for idx , path in enumerate(pathIn) : 
    if (idx % 2 == 0) | (idx % 5 == 0) :
        continue
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()


