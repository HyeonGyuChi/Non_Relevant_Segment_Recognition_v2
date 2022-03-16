



from core.util.hem.offline import *
from core.util.hem.online import *

__all__ = ['HEMHelper']


class HEMHelper():
    """
        Help computation ids for Hard Example Mining.        
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.online_helper = OnlineHEM(args)
        self.offline_helper = OfflineHEM(args)
        self.offline_methods = ['hem-softmax_diff_small-offline', 
                                'hem-softmax_diff_large-offline', 
                                'hem-voting-offline', 
                                'hem-mi_small-offline', 
                                'hem-mi_large-offline']
        
        self.set_method()
        
    def set_method(self):
        if self.args.hem_extract_mode in self.offline_methods:
            self.method = method # 'hem-vi-offline'
        else:
            self.method = method
            
    def compute_hem(self, *args):
        if 'online' in self.method:
            return self.online_helper.apply(*args)
        elif 'offline' in self.method:
            return self.offline_helper.apply(*args)
        else:
            return None