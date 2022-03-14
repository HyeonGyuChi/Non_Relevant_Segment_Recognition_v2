import torch
import torch.nn as nn


from core.util.metric import MetricHelper



class Trainer():
    def __init__(self, args):
        self.args = args
        
        
    def setup(self):
        self.current_epoch = 1
    
    
        print('======= Set Metric Helper =======')
        self.metric_helper = MetricHelper(self.args)
    
    def fit(self):    
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.args.max_epoch + 1):
            self.current_epoch = epoch
            self.metric_helper.update_epoch(epoch)
            
            # train phase
            self.train()
            
            # validation phase
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
            
            
            # TODO 고치기
            cls_hat = []
            for ti in range(len(self.n_class_list)):
                classes = torch.argmax(y_hat[ti], -1)
                cls_hat.append(classes.reshape(-1))
            
            self.metric_helper.write_preds(cls_hat, y)
            
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
        if 'online' in self.args.hem_extract_mode:
            emb, y_hat = self.model(x)
        else:
            y_hat = self.model(x)
        
        loss = self.calc_loss(y_hat, y)
        
        return y_hat, loss 
    
    
    def calc_loss(self, y_hat, y):
        # y_hat : N_task x (B x seq X C)
        # y : B x seq x (N_task classes)
        loss = 0
        loss_div_cnt = 0

        if 'ce' in self.args.loss_fn:
            for ti in range(len(self.n_class_list)):
                for seq in range(y.shape[1]):
                    loss += self.loss_fn(y_hat[ti][:, seq, ], y[:, seq, ti])
                loss_div_cnt += 1
        else: # cb, bs, eqlv2
            for ti in range(len(self.n_class_list)):
                for seq in range(y.shape[1]):
                    loss += self.loss_fn[ti](y_hat[ti][:, seq, :], y[:, seq, ti])
                loss_div_cnt += 1
                    
            if self.args.use_normsoftmax:
                for ti in range(len(self.n_class_list)):
                    loss += self.loss_fn[ti+4](y_hat[4], y[:, :, ti])   
                loss_div_cnt += 1
            
        loss /= loss_div_cnt
            
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