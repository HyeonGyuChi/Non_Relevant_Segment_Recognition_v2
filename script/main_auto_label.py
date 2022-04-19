def main():
    import os
    import torch
    from config.base_opts import parse_opts
    from core.util.database import DBHelper
    from config.meta_db_config import subset_condition
    import warnings
    from core.api.inference_autolabel import InferenceDB
    from core.api.trainer_autolabel import Trainer
    warnings.filterwarnings("ignore")

    parser = parse_opts()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list
    
    db_helper = DBHelper(args)
    db_helper.remove_table()
    db_helper.make_table()
    db_helper.random_attr_generation()
    
    # training
    autolabel_trainer = Trainer(args)
    autolabel_trainer.fit()

    # inference + make json
    args.restore_path =autolabel_trainer.args.save_path    
    infer = InferenceDB(args)
    infer.load_model()
    infer.set_inference_interval(args.inference_interval)
    results = infer.inference_new()

    #### DB UPDATE
    db_helper.update_no_anno(
        [['ANNOTATION_V1', 1],],
        "ANNOTATION_V1=False",
    )
    AFTER_df = db_helper.select(cond_info=None)
    print(AFTER_df)

    
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
    
    main()