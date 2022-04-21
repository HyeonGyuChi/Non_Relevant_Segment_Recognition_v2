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
        'SURGERY_TYPE IN ("G_1", "G_2")'
    ],
    'val': [
        'SURGERY_TYPE IN ("G_3")'
    ],
    'test': [
        'ANNOTATION_V1==0'

    ]
}