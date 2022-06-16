
def main():
    import os
    import warnings
    from config.base_opts import parse_opts
    from core.api.trainer import Trainer
    from core.api.inference import InferenceDB
    from core.api.evaluation import Evaluator
    from core.util.visualization import VisualTool

    warnings.filterwarnings("ignore")
    parser = parse_opts()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    # training
    # trainer = Trainer(args,version="WithSSIM")
    trainer = Trainer(args,version="WithoutSSIM")
    trainer.fit()
    
    # inference
    args.restore_path = trainer.args.save_path    
    infer = InferenceDB(args)
    infer.load_model()
    infer.set_inference_interval(args.inference_interval)
    results = infer.inference()
    
    # evaluation
    evaluator = Evaluator(args)
    evaluator.set_inference_interval(args.inference_interval)
    evaluator.evaluation(results_dict=results)
    
    # visualize
    visual_tool = VisualTool(args)
    visual_tool.set_result_path('{}/inference_results'.format(args.save_path))
    visual_tool.visualize()

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        print('base path : ', base_path)

    main()

