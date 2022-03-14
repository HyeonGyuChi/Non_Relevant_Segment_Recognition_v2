



from core.util.hem.offline import *
from core.util.hem.online import *






class HEMHelper():
    """
        Help computation ids for Hard Example Mining.
        
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.NON_HEM, self.HEM = (0, 1)
        self.IB_CLASS, self.OOB_CLASS = (0, 1)
        # self.bsz = self.args.batch_size # not used
        self.cnt = 0
        self.alpha = self.args.alpha
        self.con_loss = ContrastiveLoss(margin=2.0)

        self.method_idx = 0
        
        
    def set_method(self, method):
        if method in ['hem-softmax_diff_small-offline', 'hem-softmax_diff_large-offline', 'hem-voting-offline', 'hem-mi_small-offline', 'hem-mi_large-offline']:
            self.method = method # 'hem-vi-offline'
        else:
            self.method = method
            
    def compute_hem(self, *args):
        if self.method in ['hem-softmax_diff_small-offline', 'hem-softmax_diff_large-offline', 'hem-voting-offline', 'hem-mi_small-offline', 'hem-mi_large-offline']:
            return self.hem_vi_offline(*args)

        elif self.method == 'hem-emb-online':
            return self.hem_sim_method(*args)
            # if self.args.emb_type == 1 or self.args.emb_type == 2:
            #     return self.hem_cos_hard_sim(*args)
            # elif self.args.emb_type == 3:
            #     return self.hem_cos_hard_sim2(*args)
            # elif self.args.emb_type == 4:
            #     return self.hem_cos_hard_sim_only(*args)
        else: # exception
            return None