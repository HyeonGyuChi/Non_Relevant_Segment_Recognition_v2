def main():
    import os
    import warnings
    from config.base_opts import parse_opts
    warnings.filterwarnings("ignore")
    parser = parse_opts()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    from core.util.database import DBHelper
    from config.meta_db_config import subset_condition
    from core.util.compare_labels import compare_labels
    

     # gt - auto - ssim compare

     # DB에서 랜덤으로 
    # db_helper = DBHelper(args)    
    # asset_df = db_helper.select(subset_condition["compare"])
    # print("asset_df\n",asset_df)
    # SURGERY =list(asset_df['SURGERY'])
    # SURGERY_TYPE =list(asset_df['SURGERY_TYPE'])
    # PATIENT =list(asset_df['PATIENT'])
    # patient_list =[]
    # for i in range(len(PATIENT)):
    #     patient_list.append(SURGERY[i]+"/"+SURGERY_TYPE[i]+"/"+PATIENT[i])
    
    # compare_labels(patient=patient_list[10])

    # patient 입력으로 
    compare_labels(patient="robot/G_1/R_412")
    


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
    
    main()