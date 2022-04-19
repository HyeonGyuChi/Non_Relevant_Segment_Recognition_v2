import numpy as np
import sqlite3
import pandas as pd

from config.meta_db_config import *


class DBHelper():
    def __init__(self, args):
        self.args = args
        self.table_name = 'NRS'
        self.table_elements = table_elements
        self.db_path = self.args.db_path

        self.connector = sqlite3.connect(self.db_path)
        self.cursor = self.connector.cursor()

    def close_connection(self):
        self.connector.close()

    def make_table(self):
        cmd = f'CREATE TABLE IF NOT EXISTS {self.table_name}('

        for k, v in self.table_elements.items():
            cmd += '{} {},'.format(k, v)

        cmd = cmd[:-1] + ')'

        self.cursor.execute(cmd)
        self.connector.commit()

    def remove_table(self):
        cmd = f'DROP TABLE {self.table_name}'

        self.cursor.execute(cmd)
        self.connector.commit()

    def add_column(self, data):
        cmd = 'ALTER TABLE {} ADD COLUMN {} {}'.format(self.table_name, **data)
        self.cursor.execute(cmd)
        self.connector.commit()

    def select(self, cond_info):
        if cond_info is not None:
            cond = ' WHERE '
            for idx, _info in enumerate(cond_info):
                cond += '{}'.format(_info)

                if idx+1 != len(cond_info):
                    cond += ' AND '

            cmd = 'SELECT * FROM {} {}'.format(self.table_name, cond)

        else:
            cmd = 'SELECT * FROM {}'.format(self.table_name)

        return pd.read_sql_query(cmd, self.connector)
    
    def select_no_anno(self, cond_info):
        if cond_info is not None:
            cond = ' WHERE ' + cond_info
            cmd = 'SELECT * FROM {} {}'.format(self.table_name, cond)

        else:
            cmd = 'SELECT * FROM {}'.format(self.table_name)
        return pd.read_sql_query(cmd, self.connector)


    def update(self, replace_data, cond_info):
        if cond_info is not None:
            cond = 'WHERE '
            req = ''

            for idx, _info in enumerate(cond_info):
                cond += '{}'.format(_info)

                if idx+1 != len(cond_info):
                    cond += ' AND '

            for idx, _info in enumerate(replace_data):
                req += '{} = {}'.format(*_info)

                if idx+1 != len(replace_data):
                    req += ', '

            cmd = 'UPDATE {} SET {} {}'.format(self.table_name, req, cond)
        else:
            raise 'Error! not found where condition'

        self.cursor.execute(cmd)
        self.connector.commit()

    def update_no_anno(self, replace_data, cond_info):
        if cond_info is not None:
            cond = 'WHERE '+cond_info
            req = ''
            for idx, _info in enumerate(replace_data):
                req += '{} = {}'.format(*_info)

                if idx+1 != len(replace_data):
                    req += ', '
            cmd = 'UPDATE {} SET {} {}'.format(self.table_name, req, cond)
        else:
            raise 'Error! not found where condition'

        self.cursor.execute(cmd)
        self.connector.commit()

    def update_table_elements(self):
        pass
        # self.table_elements = 

    def insert(self, data):
        # table_attribute 순서에 맞게 넣어주어야 함..!
        cmd = f'INSERT INTO {self.table_name}('
        
        for idx, k in enumerate(self.table_elements.keys()):
            cmd += '{}'.format(k)

            if idx+1 != len(self.table_elements):
                cmd += ','

        cmd += ') VALUES ('
        for idx, _data in enumerate(data):
            if isinstance(_data, str):
                cmd += '"{}"'.format(_data)
            else:
                cmd += '{}'.format(_data)

            if idx+1 != len(data):
                cmd += ','

        cmd += ')'

        self.cursor.execute(cmd)
        self.connector.commit()

    def delete(self, query):
        pass
        # cmd = f'DELETE INTO {self.table_name} VALUES ("ROBOT", "GS", "R1", "01_ch1", 30.0, 25, 102058, 3038, 38.28)'

        # self.cursor.execute(cmd)
        # self.connector.commit()

    def random_attr_generation(self):
        import os
        import random
        import numpy as np

        #jw path ̰
        base_path = '../core/dataset/NRS/toyset'

        for s in os.listdir(base_path): # source?
            dpath = base_path + f'/{s}'

            for g in os.listdir(dpath): # surgery type
                dpath2 = dpath + f'/{g}'

                for p in os.listdir(dpath2): # patient
                    dpath3 = dpath2 + f'/{p}'

                    # for v in os.listdir(dpath3): # video
                    #     dpath4 = dpath3 + f'/{v}'

                    data = [s, g, p, #v, # surgery, type, patient, video
                            'GS', 'S1', # hospital, surgeon                                
                            random.choice([29.97, 30, 59.94, 60]), # fps
                            np.random.randint(38401, 1048038), # tot_frame
                            np.random.randint(38402, 1048038-38402), # rs_frame
                            np.random.randint(0, 38401), # nrs_frame

                            np.random.rand(1)[0], # rs_ratio
                            np.random.rand(1)[0], # nrs_ratio
                            np.random.randint(0, 9999), # nrs_cnt

                            np.random.randint(100000000, 999999999), # tot_duration
                            np.random.randint(100000000, 999999999), # nrs_duration
                            
                            np.random.rand(1)[0], # dup_tot_ratio
                            np.random.rand(1)[0], # dup_rs_ratio
                            np.random.rand(1)[0], # dup_nrs_ratio
                            False, # anno_v1
                            False, # anno_v2
                            False, # anno_v3
                            ]

                    self.insert(data=data)                        

