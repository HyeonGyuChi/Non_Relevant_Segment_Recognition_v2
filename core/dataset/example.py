# import json

# anno_path = "../core/dataset/NRS/lapa/vihub/anno/v3/04_GS4_99_L_10_01_NRS_30.json"

# with open(anno_path) as f: 
#     data=""
#     print("anno_path json",anno_path)
#     for line in f:            
#         data+=line
#         with open("../core/dataset/NRS/lapa/vihub/example.json", 'w') as f:
#             json.dump(data, f)
#     print(data)

import json

anno_path = "../core/dataset/NRS/lapa/vihub/anno/v3/04_GS4_99_L_69_01_NRS_30.json"

with open(anno_path) as f: 
    print("anno_path json",anno_path)
    data = ""
    for line in f:            
        print(line)
        data+=line
        
    print("data",data)



    # with open("../core/dataset/NRS/lapa/vihub/example.json", 'w') as f:
    #     json.dump(data, f)
