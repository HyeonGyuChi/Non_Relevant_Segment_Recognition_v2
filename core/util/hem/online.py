import torch
import torch.nn.functional as F



def online_method(emb, y_hat, proxies, loss_fn):
        # emb = B x ch
        # proxies = ch x classes
        if self.args.update_type2:
            classification_loss = 0
        else:
            classification_loss = loss_fn(y_hat, y)
        
        proxy_loss = 0
        
        # upate 방법 별 분기
        sim_dist = torch.zeros((emb.size(0), model.proxies.size(1))).to(emb.device)
    
        for d in range(sim_dist.size(1)):
            sim_dist[:, d] = 1 - torch.nn.functional.cosine_similarity(emb, model.proxies[:, d].unsqueeze(0))
        
        sim_preds = torch.argmin(sim_dist, -1)
        
        correct_answer = sim_preds == y
        wrong_answer = sim_preds != y
        
        # proxy all 동등 Update
        if self.args.update_type == 1:
            if self.args.use_dist_loss:
                for pi, _y in enumerate(y):
                    a = pos_emb[pi:pi+1, ]
                    b = torch.nn.functional.normalize(model.proxies[:, 0:1].transpose(0,1))
                    c = torch.nn.functional.normalize(model.proxies[:, 1:2].transpose(0,1))
                    proxy_loss += (self.con_loss(a, b, _y) + self.con_loss(a, c, _y))

                proxy_loss /= len(emb)
            else:
                proxy_loss = loss_fn(sim_dist, y)
            
            if self.args.update_type2:
                classification_loss = loss_fn(y_hat, y)
            
        # proxy correct만 update
        elif self.args.update_type == 2:
            if sum(correct_answer) > 0:
                pos_y_hat = y_hat[correct_answer]
                pos_y = y[correct_answer]
                proxy_loss = loss_fn(sim_dist[correct_answer], pos_y)
                
                if self.args.update_type2:
                    classification_loss = loss_fn(pos_y_hat, pos_y)
                
        # proxy wrong만 update (가중치 다르게)
        elif self.args.update_type == 3:
            if sum(wrong_answer) > 0:
                neg_y_hat = y_hat[wrong_answer]
                neg_y = y[wrong_answer]
                
                wrong_sim_dist = sim_dist[wrong_answer, neg_y]
                wrong_ids = torch.argsort(wrong_sim_dist)
                
                neg_sim_dist = sim_dist[wrong_ids]
                neg_y = neg_y[wrong_ids]
                
                if self.args.use_step_weight:
                    w = torch.Tensor(np.array(list(range(len(wrong_ids), 0, -1))) / len(wrong_ids)).cuda()
                    
                    for wi in range(len(wrong_sim_dist)):                        
                        n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                        proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
                        
                        if self.args.update_type2:
                            n_proxy = torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1])
                            classification_loss = classification_loss + n_proxy * w[wi:wi+1]
                else:                
                    for wi in range(len(wrong_sim_dist)):
                        w = torch.exp(self.alpha - wrong_sim_dist[wi])
                        
                        n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                        proxy_loss = proxy_loss + n_proxy * w #w[wi:wi+1]
                        
                        if self.args.update_type2:
                            n_proxy = torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1])
                            classification_loss = classification_loss + n_proxy * w # w[wi:wi+1]
                    
        # correct update + wrong update (가중치로) -> 이전 학습 세팅
        elif self.args.update_type == 4:
            if sum(correct_answer) > 0:
                pos_y_hat = y_hat[correct_answer]
                pos_y = y[correct_answer]
                
                if self.args.use_dist_loss:
                    pos_emb = emb[correct_answer]

                    for pi, _y in enumerate(pos_y):
                        a = pos_emb[pi:pi+1, ]
                        b = torch.nn.functional.normalize(model.proxies[:, 0:1].transpose(0,1))
                        c = torch.nn.functional.normalize(model.proxies[:, 1:2].transpose(0,1))
                        proxy_loss += (self.con_loss(a, b, _y) + self.con_loss(a, c, _y))

                    proxy_loss /= len(pos_emb)
                else:
                    proxy_loss = proxy_loss + loss_fn(sim_dist[correct_answer], pos_y)
                # proxy_loss = proxy_loss + loss_fn(1-sim_dist[correct_answer], pos_y)
                
                if self.args.update_type2:
                    classification_loss = loss_fn(pos_y_hat, pos_y)
            
            if sum(wrong_answer) > 0:
                neg_y_hat = y_hat[wrong_answer]
                neg_y = y[wrong_answer]
                
                wrong_sim_dist = sim_dist[wrong_answer, neg_y]
                wrong_ids = torch.argsort(wrong_sim_dist)
                
                neg_sim_dist = sim_dist[wrong_ids]
                neg_y = neg_y[wrong_ids]
                neg_emb = emb[wrong_answer]
                
                if self.args.use_step_weight:
                    w = torch.Tensor(np.array(list(range(len(wrong_ids), 0, -1))) / len(wrong_ids)).cuda()
                    
                    for wi in range(len(wrong_sim_dist)):      
                        if self.args.use_dist_loss:
                            a = neg_emb[wi:wi+1, ]
                            b = torch.nn.functional.normalize(model.proxies[:, 0:1].transpose(0,1))
                            c = torch.nn.functional.normalize(model.proxies[:, 1:2].transpose(0,1))
                            ab_loss = self.con_loss(a, b, neg_y[wi:wi+1])
                            ac_loss = self.con_loss(a, c, neg_y[wi:wi+1])                            
                            abc_loss = ab_loss + ac_loss
                            _w = w[wi]
                            abc_loss = abc_loss * _w

                            proxy_loss += abc_loss
                        else:
                            n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                            # n_proxy = torch.nn.functional.cross_entropy(1-neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                            proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
                        
                        if self.args.update_type2:
                            n_proxy = torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1])
                            # n_proxy = torch.nn.functional.cross_entropy1-(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1])
                            classification_loss = classification_loss + n_proxy * w[wi:wi+1]
                            
                    proxy_loss /= len(w)
                else:
                    for wi in range(len(wrong_sim_dist)):
                        w = torch.exp(self.alpha - wrong_sim_dist[wi])
                        
                        n_proxy = torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1])
                        # n_proxy = torch.nn.functional.cross_entropy(1-neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                        proxy_loss = proxy_loss + n_proxy * w #w[wi:wi+1]
                        
                        if self.args.update_type2:
                            n_proxy = torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1])
                            classification_loss = classification_loss + n_proxy * w #w[wi:wi+1]
                    
        # no proxy update   
        elif self.args.update_type == 5:
            if self.args.update_type2:
                classification_loss = loss_fn(y_hat, y)
            proxy_loss = 0
            
        elif self.args.update_type == 6:
            if sum(correct_answer) > 0:
                pos_y_hat = y_hat[correct_answer]
                pos_y = y[correct_answer]
                proxy_loss = proxy_loss + loss_fn(sim_dist[correct_answer], pos_y)
                
                if self.args.update_type2:
                    classification_loss = loss_fn(pos_y_hat, pos_y)
            
            if sum(wrong_answer) > 0:
                neg_y_hat = y_hat[wrong_answer]
                neg_y = y[wrong_answer]
                
                wrong_sim_dist = sim_dist[wrong_answer, neg_y]
                wrong_ids = torch.argsort(wrong_sim_dist)
                
                neg_sim_dist = sim_dist[wrong_ids]
                neg_y = neg_y[wrong_ids]
                
                if self.args.use_step_weight:
                    w = torch.Tensor(np.array(list(range(len(wrong_ids), 0, -1))) / len(wrong_ids)).cuda()
                    
                    for wi in range(len(wrong_sim_dist)):                        
                        if self.args.use_dist_loss:
                            n_proxy = wrong_sim_dist[wi]
                            if n_proxy > 0.7:
                                proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
                        else:
                            n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                            proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
                        
                        if self.args.update_type2:
                            n_proxy = torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1])
                            classification_loss = classification_loss + n_proxy * w[wi:wi+1]
                            
        if classification_loss + proxy_loss == 0:
            classification_loss = torch.tensor([0.0], requires_grad=True).cuda()
            
        return classification_loss + proxy_loss