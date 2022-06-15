# Table elements
# TEXT : string
# INTEGER : int type
# REAL : float or double
# BOOL : 0 or 1

table_elements = {
    'SURGERY': 'TEXT',
    'SURGERY_TYPE': 'TEXT',
    'PATIENT': 'TEXT',
    # 'VIDEO': 'TEXT',
    'HOSPITAL': 'TEXT',
    'SURGEON': 'TEXT',

    'FPS': 'REAL',
    'TOT_FRAME': 'INTERGER',
    'RS_FRAME': 'INTERGER',
    'NRS_FRAME': 'INTERGER',

    'RS_RATIO': 'REAL',
    'NRS_RATIO': 'REAL',
    'NRS_CNT': 'INTEGER',

    'TOT_DURATION': 'INTEGER',
    'NRS_DURATION': 'INTEGER',

    'DUP_TOT_RATIO': 'REAL',
    'DUP_RS_RATIO': 'REAL',
    'DUP_NRS_RATIO': 'REAL',

    'ANNOTATION_V1': 'BOOL',
    'ANNOTATION_V2': 'BOOL',
    'ANNOTATION_V3': 'BOOL', 
}


# subset config
# 특정 조건들... (nrs ratio 등 해당 조건에 맞는 데이터를 불러오도록 하기)
subset_condition = {
    'train': [
        'SURGERY== "lapa" and PATIENT in ("01_VIHUB1.2_B4_L_150","04_GS4_99_L_102","04_GS4_99_L_29","04_GS4_99_L_5","04_GS4_99_L_91","01_VIHUB1.2_A9_L_51","04_GS4_99_L_64")'

    ],
    'val': [
         'SURGERY== "lapa" and PATIENT in ("04_GS4_99_L_97","01_VIHUB1.2_B4_L_2")',
        
    ],
    'test': [
        'PATIENT == "01_VIHUB1.2_B4_L_151"',
        # 'ANNOTATION_V1==False'
    ],
    'ssim': [
        'PATIENT == "04_GS4_99_L_5"'
        # 'ANNOTATION_V3==False'
    ],
    'compare': [
        'ANNOTATION_V3==True'
    ],
}