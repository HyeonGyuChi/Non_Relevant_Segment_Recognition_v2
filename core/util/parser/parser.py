import os
import json
import pandas as pd
import numpy as np
import yaml
import re

class FileLoader():
    def __init__(self, file_path=''):
        import os
        import json
        import pandas as pd
        import numpy as np
        import yaml
        import re
        self.file_path = file_path

    def set_file_path(self, file_path):
        self.file_path = file_path

    def get_full_path(self):
        return os.path.abspath(self.file_path)

    def get_file_name(self):
        return os.path.splitext(self.get_basename())[0]

    def get_file_ext(self):
        return os.path.splitext(self.get_basename())[1]
    
    def get_basename(self):
        return os.path.basename(self.file_path)
    
    def get_dirname(self):
        return os.path.dirname(self.file_path)
    
    def load(self):
        # https://stackoverflow.com/questions/9168340/using-a-dictionary-to-select-function-to-execute
        support_loader = {
            '.json':(lambda x: self.load_json()), # json object
            '.csv':(lambda x: self.load_csv()), # Dataframe
            '.yaml':(lambda x: self.load_yaml()), # dict
            '.png':-1 # PIL
        }

        data = support_loader.get(self.get_file_ext(), -1)('dummy')

        # assert data_loader != -1, 'NOT SUPPOERT FILE EXTENSION ON FileLoader'

        return data

    def load_json(self): # to json object
        with open(self.file_path) as self.json_file :
            return json.load(self.json_file)

    def load_csv(self): # to Dataframe
        df = pd.read_csv(self.file_path)
        return df
    
    def load_yaml(self): # to dict
        load_dict = {}

        with open(self.file_path, 'r') as f :
            load_dict = yaml.load(f, Loader=yaml.FullLoader)
    
        return load_dict