import torch
import torch.nn as nn


from core.util.hem import HEMHelper
from core.util.metric import MetricHelper



class Trainer():
    def __init__(self, args):
        self.args = args
        
        
    def setup(self):
        self.current_epoch = 1
        self.current_state = 'train' # base set
    
        print('======= Set HEM Helper =======')
        self.hem_helper = HEMHelper(self.args)
    
        print('======= Set Metric Helper =======')
        self.metric_helper = MetricHelper(self.args)
    
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
        
        for data in tqdm(self.train_loader, desc='[Train Phase] : '):
            self.optimizer.zero_grad()
            
            x, y = data
            if self.args.device == 'cuda':
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                
            y_hat, loss = self.forward(x, y)
            
            self.metric_helper.write_loss(loss.item(), 'train')
            
            loss.backward()
            self.optimizer.step()
            
        self.metric_helper.update_loss('train')
    
    @torch.no_grad()
    def valid(self):
        self.model.eval()

        for data in tqdm(self.val_loader, desc='[Validation Phase] : '):
            x, y = data
            
            if self.args.device == 'cuda':
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                
            y_hat, loss = self.forward(x, y)
            
            self.metric_helper.write_loss(loss.item(), 'valid')
            self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu()) # MetricHelper 에 저장
            
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
        loss = self.calc_loss(outputs, y)
        
        return y_hat, loss 
    
    
    def calc_loss(self, outputs, y):
        if 'online' in self.args.hem_extract_mode and self.current_state = 'train':
            emb, y_hat = outputs
            loss = self.hem_helper.compute_hem(self.model, x, y, self.loss_fn)
        else:
            y_hat = outputs
            loss = self.loss_fn(y_hat, y)
            
        return loss
    
    def save_checkpoint(self):
        saved_pt_list = glob(os.path.join(self.args.save_path, '*pth'))

        if len(saved_pt_list) > self.args.save_top_n:
            saved_pt_list = natsort.natsorted(saved_pt_list)

            for li in saved_pt_list[:-(self.args.save_top_n+1)]:
                os.remove(li)

        save_path = '{}/epoch:{}-{}:{:.4f}.pth'.format(
                    self.args.save_path,
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

        print('[+] save checkpoint : ', save_path)