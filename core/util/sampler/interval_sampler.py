from torch.utils.data import Sampler


class IntervalSampler(Sampler):
    
    def __init__(self, data_source, interval):
        self.data_source = data_source
        self.interval = interval
    
    def __iter__(self):
        return iter(range(0, len(self.data_source), self.interval))
    
    def __len__(self):
        return len(self.data_source)