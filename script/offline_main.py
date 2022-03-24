
def main():
    import copy
    from config.base_opts import parse_opts
    from core.api.trainer import Trainer
    from core.api.inference import InferenceDB
    from core.api.evaluation import Evaluator
    from core.util.hem import OfflineHEM
    
    
    parser = parse_opts()
    args = parser.parse_args()
    args_bak = copy.deepcopy(args)
        
    for cur_stage in range(1, args.n_stage+1):
        if args.appointment_assets_path != '':
            args_bak.appointment_assets_path = args.appointment_assets_path
        
        args_bak.cur_stage = cur_stage
        
        # train mini folds
        # for train_stage in range(args.n_mini_fold):
        #     args = copy.deepcopy(args_bak)
        #     args.train_stage = 'mini_fold_stage_{}'.format(train_stage)
        
        #     # training
        #     trainer = Trainer(args)
        #     trainer.fit()
        
        #     # # inference
        #     # args.train_stage = 'general_train'
        #     args.restore_path = trainer.args.save_path
        
        #     infer = InferenceDB(args)
        #     infer.load_model()
        #     infer.set_inference_interval(args.inference_interval)
        #     results = infer.inference()
            
        #     evaluator = Evaluator(args)
        #     evaluator.set_inference_interval(args.inference_interval)
        #     evaluator.evaluation(results_dict=results)
            
        # extract hem asset
        args = copy.deepcopy(args_bak)
        # args.sample_ratio = 15 # set 5 fps
        # args.sample_type = 'all' # change to all mode
        
        # extractor = OfflineHEM(args)
        # extractor.extract(method_name='hem-softmax_diff_small-offline')
        # extractor.aggregate_assets(method_name='hem-softmax_diff_small-offline')
        # args.appointment_assets_path = extractor.get_aggregate_path(method_name='hem-softmax_diff_small-offline')
        
        args.appointment_assets_path = './logs/mobilenetv3_large_100-1-mini_fold_stage_0-offline-1/hem_assets/hem-softmax_diff_small-offline-agg.csv'
        
        # train model for hem asset
        args.train_stage = 'general_train'
        args.sample_ratio = 30 # set 5 fps
        args.sample_type = 'boundary' # change to all mode
        
        trainer = Trainer(args)
        trainer.fit()
        
    
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        
        print('base path : ', base_path)

    main()

