import os
import sys
import random
import numpy as np
import torch
from glob import glob
from PIL import Image
import pandas as pd
import natsort
from torch.utils.data import Dataset

from core.dataset.fold_robot import *
from core.dataset.aug_info import *



class RobotDataset(Dataset):
    def __init__(self, args, state='train'):
        super().__init__()
        
        self.args = args
        self.state = state
        self.fold = self.args.fold
        
        train_videos, val_videos = robot_train_videos, robot_val_videos
        
        
        self.aug = d_transforms[self.state]
        
    
        

