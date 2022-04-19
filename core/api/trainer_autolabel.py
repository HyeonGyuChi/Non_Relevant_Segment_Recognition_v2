import os
import json
from tqdm import tqdm
from glob import glob
import natsort
import torch
import torch.nn as nn
from core.model import get_model, get_loss, configure_optimizer
from core.dataset import load_data
# from core.util.hem import HEMHelper, OnlineHEM
from core.util.hem import OnlineHEM
from core.util.metric import MetricHelper
from torch.utils.data import DataLoader
from core.dataset import SubDataset

import torchvision
from torchvision.models import resnet18

def pretrained_resnet50():
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        return model

class Trainer():
    def __init__(self, args):
        self.args = args        
        self.setup()
    
    def setup(self):
        self.current_epoch = 1
        self.current_state = 'train' # base set
        
        # make log directory
        self.make_log_dir()
        self.save_hyperparams()
    
        print('======= Load model =======')
        self.model = get_model(self.args).to(self.args.device)

        print('======= Load loss =======')
        self.loss_fn = get_loss(self.args)
        
        print('======= Load Optimizers =======')
        self.optimizer, self.scheduler = configure_optimizer(self.args, self.model)
    
        print('======= Load dataset =======')
        trainset = SubDataset(self.args, state='train', sample_type=self.args.sample_type)
        valset   = SubDataset(self.args, state='val', sample_type=self.args.sample_type)
        
        if self.args.sampler == 'oversampler':
            self.train_loader = DataLoader(trainset,
                                    batch_size=self.args.batch_size,
                                    num_workers=self.args.num_workers,
                                    sampler=OverSampler(trainset.label_list, self.args.batch_size//2, self.args.batch_size),
                                    pin_memory=True,
                                    )
        else:
            self.train_loader = DataLoader(trainset,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers,
                                shuffle=True,
                                pin_memory=True,
                                )
        
        self.val_loader = DataLoader(valset,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers,
                                shuffle=False,
                                pin_memory=True,
                                )
        print("Train Data:        ",len(self.train_loader), "\nValidation Data:",len(self.val_loader)) 

        print('======= Set HEM Helper =======')
        self.hem_helper = OnlineHEM(self.args)
    
        print('======= Set Metric Helper =======\n\n')
        self.metric_helper = MetricHelper(self.args)
           
    def make_log_dir(self):
        if 'mini' in self.args.train_stage:
            self.args.save_path += '/{}-{}-{}-{}-1'.format(self.args.model,
                                                        self.args.cur_stage,
                                                        self.args.train_stage,
                                                       self.args.hem_extract_mode)
        elif self.args.appointment_assets_path != '':
            self.args.save_path += '/{}-{}-hem-asset-train-1'.format(self.args.model,
                                                        self.args.cur_stage,)
        else:
            self.args.save_path += '/{}-{}-{}-1'.format(self.args.model,
                                                        self.args.train_stage,
                                                       self.args.hem_extract_mode)
        
        # 같은 이름의 로그 디렉토리 중복 회피
        if os.path.exists(self.args.save_path):
            for idx in range(2, 99):
                tmp_save_path = self.args.save_path[:-2] + '-{}'.format(idx)
                
                if not os.path.exists(tmp_save_path):
                    self.args.save_path = tmp_save_path
                    break
                
        os.makedirs(self.args.save_path, exist_ok=True)
        print('Make log dir : ', self.args.save_path)
    
    def save_hyperparams(self):
        save_path = self.args.save_path + '/hparams.yaml'
            
        hparams = vars(self.args)
        with open(save_path, 'w') as f:
            json.dump(hparams, f, indent=2)
        
        print('[+] save hyperparams : {}\n'.format(save_path))
        
    def fit(self):    
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.args.max_epoch + 1):
            self.current_epoch = epoch
            self.metric_helper.update_epoch(epoch)
            
            # train phase
            self.current_state = 'train'
            self.train()
            
            # validation phase
            self.current_state = 'val'
            self.valid()
        
    def train(self):
        self.model.train()
        cnt = 0
        for data in tqdm(self.train_loader, desc='[Train Phase] : '):
            self.optimizer.zero_grad()
            
            _, x, y = data
            if self.args.device == 'cuda':
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                
            y_hat, loss = self.forward(x, y)
            
            self.metric_helper.write_loss(loss.item(), 'train')
            
            loss.backward()
            self.optimizer.step()
            
            cnt += 1
            if cnt > 3:
                break
            
        self.metric_helper.update_loss('train')
    
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        cnt = 0

        for data in tqdm(self.val_loader, desc='[Validation Phase] : '):
            _, x, y = data
            
            if self.args.device == 'cuda':
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                
            y_hat, loss = self.forward(x, y)
            
            self.metric_helper.write_loss(loss.item(), 'valid')
            self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu()) # MetricHelper 에 저장
            
            cnt += 1
            if cnt > 3:
                break
            
        self.metric_helper.update_loss('valid')
        self.metric_helper.save_loss_pic()
        metric = self.metric_helper.calc_metric()
        
        if self.args.lr_scheduler == 'reduced':
            self.scheduler.step(self.metric_helper.get_loss('valid'))
        else:
            self.scheduler.step()
        
        if self.metric_helper.update_best_metric(metric):
            self.save_checkpoint()
        
    def forward(self, x, y):
        outputs = self.model(x)
        return self.calc_loss(outputs, y)
    
    def calc_loss(self, outputs, y):
        if 'online' in self.args.hem_extract_mode and self.current_state == 'train':
            emb, y_hat = outputs
            loss = self.hem_helper.apply(emb, y_hat, y, self.model.proxies)
        else:
            y_hat = outputs
            loss = self.loss_fn(y_hat, y) 
        return y_hat, loss
    
    def save_checkpoint(self):
        ckpt_save_path = self.args.save_path + '/checkpoints'
        os.makedirs(ckpt_save_path, exist_ok=True)
        
        saved_pt_list = glob(os.path.join(ckpt_save_path, '*pth'))

        if len(saved_pt_list) > self.args.save_top_n:
            saved_pt_list = natsort.natsorted(saved_pt_list)

            for li in saved_pt_list[:-(self.args.save_top_n+1)]:
                os.remove(li)

        save_path = '{}/epoch:{}-{}:{:.4f}-best.pth'.format(
                    ckpt_save_path,
                    self.current_epoch,
                    self.args.target_metric,
                    self.metric_helper.get_best_metric(),
                )

        if self.args.num_gpus > 1:
            ckpt_state = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.module.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.current_epoch,
            }
        else:
            ckpt_state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.current_epoch,
            }

        torch.save(ckpt_state, save_path)
        print('[+] save checkpoint (Best Metric) : ', save_path)
        
        if self.current_epoch == self.args.max_epoch: # last epoch checkpoint saving
            save_path = '{}/epoch:{}-{}:{:.4f}-last.pth'.format(
                        ckpt_save_path,
                        self.current_epoch,
                        self.args.target_metric,
                        self.metric_helper.get_best_metric(),
                    )

            if self.args.num_gpus > 1:
                ckpt_state = {
                    'model': self.model.module.state_dict(),
                    'optimizer': self.optimizer.module.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': self.current_epoch,
                }
            else:
                ckpt_state = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': self.current_epoch,
                }

            torch.save(ckpt_state, save_path)
            print('[+] save checkpoint (Last Epoch) : ', save_path)
            
            
