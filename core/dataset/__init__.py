from torch.utils.data import DataLoader
from core.util.data.sampler import OverSampler
from core.dataset.robot_dataset import RobotDataset
# from core.dataset.lapa_dataset import LapaDataset


__all__ = ['load_data']


def load_data(args):
    if args.dataset == 'robot':
        trainset = RobotDataset(args, state='train', sample_type=args.sample_type)
        valset = RobotDataset(args, state='val', sample_type=args.sample_type)
    elif args.dataset == 'lapa':
        pass
    
    if args.sampler == 'oversampler':
        train_loader = DataLoader(trainset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sampler=OverSampler(trainset.label_list, args.batch_size//2, args.batch_size)
                                )
    else:
        train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              )
    
    val_loader = DataLoader(valset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=False,
                              )
    
    return train_loader, val_loader