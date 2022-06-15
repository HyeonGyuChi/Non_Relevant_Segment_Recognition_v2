import os
import os
from PIL import Image
from torch.utils.data import Dataset
from core.dataset.aug_info import *


class InferDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.img_list = None
        
        if self.args.experiment_type == 'ours':
            if self.args.model == 'mobile_vit':
                d_transforms = data_transforms_mvit
            else:    
                d_transforms = data_transforms
                
        elif self.args.experiment_type == 'theator':
            d_transforms = data_transforms_theator

        self.aug = d_transforms['test']
        
    def set_img_list(self, img_list):
        self.img_list = img_list
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]

        img = Image.open(img_path)
        img = self.aug(img)

        # parsing DB img idx
        return {'img': img,
                'db_idx': img_path.split('/')[-1][:-4],
                'img_path': img_path}