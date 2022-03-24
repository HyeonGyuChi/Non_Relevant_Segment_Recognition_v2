import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.model.losses import ContrastiveLoss


class OnlineHEM():
    def __init__(self, args):
        self.args = args
        self.alpha = self.args.alpha
        self.w_lambda = self.args.w_lambda
        self.ce_loss = nn.CrossEntropyLoss()
        self.con_loss = ContrastiveLoss(margin=self.args.margin)
        
        
    def apply(self, emb, y_hat, y, proxies):
        # upate 방법 별 분기
        sim_mat = torch.zeros((emb.size(0), proxies.size(1))).to(emb.device)
    
        for d in range(sim_mat.size(1)):
            sim_mat[:, d] = 1 - torch.nn.functional.cosine_similarity(emb, proxies[:, d].unsqueeze(0))
        
        sim_preds = torch.argmin(sim_mat, -1)        
        correct_answer = sim_preds == y
        wrong_answer = sim_preds != y
    
        # emb = B x ch
        # proxies = ch x classes
        proxy_loss = 0
        
        if self.args.online_type == 1: # classification은 전체 통합 업데이트
            classification_loss = self.ce_loss(y_hat, y)
            
        elif self.args.online_type == 2: # classification도 따로 업데이트
            classification_loss = 0
        
        
        if sum(correct_answer) > 0:
            pos_y_hat = y_hat[correct_answer]
            pos_y = y[correct_answer]
            pos_sim_mat = sim_mat[correct_answer]
            
            if self.args.use_contrastive:
                pos_emb = emb[correct_answer]
            else:
                pos_emb = None
                
            classification_loss2, proxy_loss2 = self.compute_non_hard(proxies, pos_sim_mat, pos_y_hat, pos_y, pos_emb)
            classification_loss = classification_loss + classification_loss2
            proxy_loss = proxy_loss + proxy_loss2
        
        if sum(wrong_answer) > 0:
            neg_y_hat = y_hat[wrong_answer]
            neg_y = y[wrong_answer]
            
            wrong_sim_mat = sim_mat[wrong_answer, neg_y]
            wrong_ids = torch.argsort(wrong_sim_mat)
            
            neg_sim_mat = sim_mat[wrong_ids]
            neg_y = neg_y[wrong_ids]
            
            if self.args.use_contrastive:
                neg_emb = emb[wrong_answer]
            else:
                neg_emb = None
            
            classification_loss3, proxy_loss3 = self.compute_hard(proxies, wrong_sim_mat, neg_sim_mat, neg_y_hat, neg_y, neg_emb)
            classification_loss = classification_loss + classification_loss3
            proxy_loss = proxy_loss + proxy_loss3
                                    
        if classification_loss == 0:
            classification_loss = torch.tensor([0.0], requires_grad=True).to(self.args.device)
            
        if proxy_loss == 0:
            proxy_loss = torch.tensor([0.0], requires_grad=True).to(self.args.device)
                
        return classification_loss + proxy_loss * self.w_lambda
                
    def compute_non_hard(self, proxies, pos_sim_mat, pos_y_hat, pos_y, pos_emb):
        proxy_loss = 0
        classification_loss = 0
            
        if pos_emb is not None:
            b = F.normalize(proxies[:, 0:1].transpose(0,1))
            c = F.normalize(proxies[:, 1:2].transpose(0,1))

            for pi, _y in enumerate(pos_y):
                a = pos_emb[pi:pi+1, ]
                proxy_loss += (self.con_loss(a, b, _y) + self.con_loss(a, c, _y))

            proxy_loss /= len(pos_emb)
        else:
            proxy_loss = proxy_loss + self.ce_loss(pos_sim_mat, pos_y)
        
        if self.args.online_type == 2:
            classification_loss = self.ce_loss(pos_y_hat, pos_y)
                
                
        return classification_loss, proxy_loss
        
    def compute_hard(self, proxies, wrong_sim_mat, neg_sim_mat, neg_y_hat, neg_y, neg_emb):  
        proxy_loss = 0
        classification_loss = 0        
        
        if self.args.use_step_weight:
            w = torch.Tensor(np.array(list(range(len(neg_y), 0, -1))) / len(neg_y)).to(self.args.device)
        else:
            w = torch.stack([torch.exp(self.alpha - wrong_sim_mat[wi]) for wi in range(len(wrong_sim_mat))], 0).to(self.args.device)
            
        if neg_emb is not None:
            b = torch.nn.functional.normalize(proxies[:, 0:1].transpose(0,1))
            c = torch.nn.functional.normalize(proxies[:, 1:2].transpose(0,1))
            
        for wi in range(len(wrong_sim_mat)):
            if neg_emb is not None:            
                a = neg_emb[wi:wi+1, ]                
                abc_loss = self.con_loss(a, b, neg_y[wi:wi+1]) + self.con_loss(a, c, neg_y[wi:wi+1])
                _w = w[wi]
                abc_loss = abc_loss * _w

                proxy_loss += abc_loss
            else:
                n_proxy = self.ce_loss(neg_sim_mat[wi:wi+1, ], neg_y[wi:wi+1])
                proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
                
                if self.args.online_type == 2:
                    n_proxy = self.ce_loss(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1])
                    classification_loss = classification_loss + n_proxy * w[wi:wi+1]
                    
            proxy_loss /= len(w)
            
        return classification_loss, proxy_loss