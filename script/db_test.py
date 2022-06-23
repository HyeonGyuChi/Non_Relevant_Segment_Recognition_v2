


def main():
    from config.base_opts import parse_opts
    from core.util.database import DBHelper
    from config.meta_db_config import subset_condition
    from core.dataset import SubDataset

    parser = parse_opts()
    args = parser.parse_args()

    db_helper = DBHelper(args)
    db_helper.remove_table()
    db_helper.make_table()
    db_helper.random_attr_generation()
    df = db_helper.select(cond_info=None)
    
    # print(df.head())

    df = db_helper.select(cond_info=subset_condition['train'])

    print(df.head())

    db_helper.update(
        [['ANNOTATION_V3', 1],],
        subset_condition['train'],
        
    )

    df = db_helper.select(cond_info=subset_condition['train'])

    print(df.head())


    dset = SubDataset(args, sample_type='boundary')

    dset = SubDataset(args, sample_type='all')
    
    
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
    
    main()