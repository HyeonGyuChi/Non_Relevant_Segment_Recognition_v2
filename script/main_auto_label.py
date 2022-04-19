def main():
    import os
    from config.base_opts import parse_opts
    from core.util.database import DBHelper
    from config.meta_db_config import subset_condition
    import warnings
    from core.api.inference import new_data_inference
    import torch
    warnings.filterwarnings("ignore")

    parser = parse_opts()
    args = parser.parse_args()
    db_helper = DBHelper(args)
    db_helper.remove_table()
    db_helper.make_table()
    db_helper.random_attr_generation()
    
    #### NEW DATA CHECK
    BEFORE_df=db_helper.select_no_anno(cond_info="ANNOTATION_V1=False")
    print(BEFORE_df)

    
    new_patient_list=BEFORE_df['PATIENT'].tolist()
    new_patient_path_list=[]
    for i in range(len(BEFORE_df['PATIENT'].tolist())):
        new_patient_path = BEFORE_df['SURGERY'][i]+"/"+BEFORE_df['SURGERY_TYPE'][i]+"/"+BEFORE_df['PATIENT'][i]
        new_patient_path_list.append(new_patient_path)

    #### NEW DATA INFERENCE
    new_data_inference(new_patient_path_list)

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