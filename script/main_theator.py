
def main():
    import os
    import copy
    import torch
    import pandas as pd
    from tqdm import tqdm
    from config.base_opts import parse_opts
    from core.api.trainer import Trainer
    from core.dataset import load_data
    from core.api.inference import InferenceDB
    from core.api.evaluation import Evaluator
    from core.util.visualization import VisualTool
    
    
    parser = parse_opts()
    args = parser.parse_args()
    args_bak = copy.deepcopy(args)

    # load base laoder
    args.use_oversample = False
    train_loader, _ = load_data(args)
        
    for cur_stage in range(1, args.n_stage+1):
        args = copy.deepcopy(args_bak)
        args.use_oversample = True
        args.cur_stage = cur_stage
        
        # training
        trainer = Trainer(args)
        trainer.fit()
    
        # inference
        args.restore_path = trainer.args.save_path
    
        infer = InferenceDB(args)
        infer.load_model()
        infer.set_inference_interval(args.inference_interval)
        results = infer.inference()
        
        evaluator = Evaluator(args)
        evaluator.set_inference_interval(args.inference_interval)
        evaluator.evaluation(results_dict=results)

        # visualize
        # visual_tool = VisualTool(args)
        # visual_tool.set_result_path('{}/inference_results'.format(args.save_path))
        # visual_tool.visualize()
        
        change_list = []
        model = trainer.model
        model.eval()

        with torch.no_grad():
            for _, img, lbs in tqdm(train_loader):
                img = img.cuda()
                outputs = model(img)
                ids = list(torch.argmax(outputs, -1).cpu().data.numpy())
                change_list += ids

        trainer.train_loader.dataset.label_list = change_list
    
        # save predict list to csv
        hem_assets_path = os.path.join(args.restore_path, 'stage_{}.csv'.format(cur_stage))
        predict_df = pd.DataFrame({
                        'img_path': train_loader.dataset.img_list,
                        'class_idx': change_list,
                    })
        predict_df.to_csv(hem_assets_path)

    
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        
        print('base path : ', base_path)

    main()

