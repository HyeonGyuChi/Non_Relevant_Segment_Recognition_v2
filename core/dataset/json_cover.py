import json

anno_path= "./NRS/lapa/vihub/anno/v3/04_GS4_99_L_68_01_NRS_30_new.json"

L_68_01_NRS_30 = {
  "totalFrame": 106021,
  "frameRate": 30,
  "width": 1920,
  "height": 1080,
  "_id": "61afe4e12bd5d3001b3578ae",
  "annotations": [
    {
      "start": 17995,
      "end": 18451,
      "code": 1,
      
    },
    {
      "start": 18591,
      "end": 19092,
      "code": 1,
      
    },
    {
      "start": 46481,
      "end": 46789,
      "code": 1,
      
    },
     {
      "start": 46834,
      "end": 47353,
      "code": 1,
      
    },
    {
      "start": 47556,
      "end": 47631,
      "code": 1,
      
    },

    {
      "start": 47758,
      "end": 47837,
      "code": 1,
      
    },
    {
      "start": 48408,
      "end": 48819,
      "code": 1,
      
    },
    {
      "start": 49374,
      "end": 49494,
      "code": 1,
      
    },
     {
      "start": 82184,
      "end": 82647,
      "code": 1,
      
    },
    {
      "start": 101626,
      "end": 101629,
      "code": 1,
      
    },
    {
      "start": 101835,
      "end": 102173,
      "code": 1,
      
    },
     {
      "start": 104064,
      "end": 106021,
      "code": 1,
      
    },
    

  ],
  "annotationType": "NRS",
  "createdAt": "2021-12-07T22:49:05.716Z",
  "updatedAt": "2022-02-03T06:24:19.402Z",
  "annotator": "30",
  "name": "04_GS4_99_L_68_01",
  "label": {
    "1": "NonRelevantSurgery"
  }
}


with open(anno_path, 'w') as f:
    json.dump(L_68_01_NRS_30, f)


# import json

# anno_path = "../core/dataset/NRS/lapa/vihub/anno/v3/04_GS4_99_L_69_01_NRS_30.json"

# with open(anno_path) as f: 
#     print("anno_path json",anno_path)
#     data = ""
#     for line in f:            
#         print(line)
#         data+=line
        
#     print("data",data)

