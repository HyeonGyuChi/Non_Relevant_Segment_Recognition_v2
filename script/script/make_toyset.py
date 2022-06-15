import os
import shutil
from natsort import natsorted
from glob import glob
from tqdm import tqdm

base_path = '/dataset/NRS'

spath = base_path + '/robot/mola/'
spath2 = spath + '/img'
spath3 = spath + '/anno/v3'

anno_list = os.listdir(spath3)

p_dict = {}

for patient in os.listdir(spath2)[:30]:
    p_dict[patient] = []

    spath22 = spath2 + f'/{patient}'

    for video in os.listdir(spath22):
        for anno in anno_list:
            if video in anno:
                p_dict[patient].append([video, anno])
                break


tpath = base_path + '/toyset/robot'
key_list = list(p_dict.keys())

for i in range(5):
    st, ed = i*6, (i+1)*6
    tpath2 = tpath + '/G_{}'.format(i+1) # surgery type no

    for k in tqdm(key_list[st:ed], desc='patient..'):
        tpath3 = tpath2 + '/{}'.format(k)

        for v in tqdm(p_dict[k], desc='video..'):
            tpath4 = tpath3 + '/{}/img'.format(v[0])
            tpath5 = tpath3 + '/{}/anno/org'.format(v[0])

            os.makedirs(tpath4, exist_ok=True)
            os.makedirs(tpath5, exist_ok=True)

            # copy
            spath22 = spath2 + '/{}/{}'.format(k, v[0])
            spath33 = spath3 + '/{}'.format(v[1])

            for fname in os.listdir(spath22):
                spath222 = spath22 + f'/{fname}'
                shutil.copy(spath222, tpath4 + '/{}'.format(fname))

            shutil.copy(spath33, tpath5 + '/{}'.format(v[1]))



spath = base_path + '/lapa/vihub_copy/'
spath2 = spath + '/img'
spath3 = spath + '/anno/v3'

anno_list = os.listdir(spath3)

p_dict = {}

for patient in os.listdir(spath2)[:30]:
    p_dict[patient] = []

    spath22 = spath2 + f'/{patient}'

    for video in os.listdir(spath22):
        for anno in anno_list:
            if video in anno:
                p_dict[patient].append([video, anno])
                break


tpath = base_path + '/toyset/lapa'
key_list = list(p_dict.keys())

for i in range(5):
    st, ed = i*6, (i+1)*6
    tpath2 = tpath + '/G_{}'.format(i+1) # surgery type no

    for k in tqdm(key_list[st:ed], desc='patient..'):
        tpath3 = tpath2 + '/{}'.format(k)

        for v in tqdm(p_dict[k], desc='video..'):
            tpath4 = tpath3 + '/{}/img'.format(v[0])
            tpath5 = tpath3 + '/{}/anno/org'.format(v[0])

            os.makedirs(tpath4, exist_ok=True)
            os.makedirs(tpath5, exist_ok=True)

            # copy
            spath22 = spath2 + '/{}/{}'.format(k, v[0])
            spath33 = spath3 + '/{}'.format(v[1])

            for fname in os.listdir(spath22):
                spath222 = spath22 + f'/{fname}'
                shutil.copy(spath222, tpath4 + '/{}'.format(fname))

            shutil.copy(spath33, tpath5 + '/{}'.format(v[1]))


