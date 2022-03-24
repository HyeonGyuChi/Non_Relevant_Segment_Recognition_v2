
def main():
    from config.base_opts import parse_opts
    from core.api.trainer import Trainer
    from core.api.inference import InferenceDB
    from core.api.evaluation import Evaluator
    from core.util.parser import AssetParser
    
    parser = parse_opts()
    args = parser.parse_args()
    
    # training
    trainer = Trainer(args)
    trainer.fit()
    
    # # inference
    args.restore_path = trainer.args.save_path
    
    infer = InferenceDB(args)
    infer.load_model()
    infer.set_inference_interval(args.inference_interval)
    results = infer.inference()
    
    evaluator = Evaluator(args)
    evaluator.set_inference_interval(args.inference_interval)
    evaluator.evaluation(results_dict=results)
    
    
    # TODO 여기서 한 번에 하는게 맞을까..?
    # visualization per patients
    # patient_predict_visual_path = os.path.join(each_patients_save_dir, 'predict-{}.png'.format(patient_no))

    # visual_tool = VisualTool(patient_gt_list, patient_no, patient_predict_visual_path)
    # visual_tool.visual_predict(patient_predict_list, self.model, self.inference_interval, window_size=300, section_num=2)


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        
        print('base path : ', base_path)

    main()

