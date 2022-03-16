


def main():
    from config.base_opts import parse_opts
    from core.api.trainer import Trainer
    
    parser = parse_opts()
    args = parser.parse_args()
    
    trainer = Trainer(args)
    


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        
        print('base path : ', base_path)

    main()

