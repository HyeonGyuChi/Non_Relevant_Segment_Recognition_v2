
def main():
    from config.base_opts import parse_opts
    from core.api.trainer import Trainer
    from core.api.inference import InferenceDB
    from core.api.evaluation import Evaluator
    from core.util.visualization import VisualTool
    
    parser = parse_opts()
    args = parser.parse_args()
    
    # training
    trainer = Trainer(args)
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

