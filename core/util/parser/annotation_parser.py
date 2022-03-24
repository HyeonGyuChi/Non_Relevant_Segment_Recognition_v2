import numpy as np
from core.util.parser.file_loader import FileLoader


class AnnotationParser():
    def __init__(self, annotation_path:str):
        self.IB_CLASS, self.OOB_CLASS = (0,1)
        self.annotation_path = annotation_path
        print('annotation_path : ', annotation_path)
        self.json_data = FileLoader(annotation_path).load()
    
    def set_annotation_path(self, annotation_path):
        self.annotation_path = annotation_path
        self.json_data = FileLoader(annotation_path).load()

    def get_totalFrame(self):
        return self.json_data['totalFrame'] # str

    def get_fps(self):
        return self.json_data['frameRate'] # float
    
    def get_annotations_info(self):
        # annotation frame
        annotation_idx_list = []
        
        for anno_data in self.json_data['annotations'] :
            start = anno_data['start'] # start frame number
            end = anno_data['end'] # end frame number

            annotation_idx_list.append([start, end]) # annotation_idx_list = [[start, end], [start, end]..]

        return annotation_idx_list

    def get_event_sequence(self, extract_interval=1):

        event_sequence = np.array([self.IB_CLASS]*self.get_totalFrame())
        annotation_idx_list = self.get_annotations_info()
        
        for start, end in annotation_idx_list:
            event_sequence[start: end+1] = self.OOB_CLASS
        
        return event_sequence.tolist()[::extract_interval]