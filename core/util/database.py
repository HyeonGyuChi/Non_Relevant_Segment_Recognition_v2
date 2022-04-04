import numpy as np
import sqlite3
import pandas as pd


class DBHelper():
    def __init__(self, args):
        self.args = args
        self.db_path = self.args.db_path

        self.connector = sqlite3.connect(self.db_path)
        self.cursor = self.connector.cursor()

        # self.connector.close()

    def make_table(self):
        table_elements = {
            'SURGERY': 'TEXT',
            'SURGERY_TYPE': 'TEXT',
            'PATIENT': 'TEXT',
            'VIDEO': 'TEXT',
            'FPS': 'REAL',
            'NRS_CNT': 'INTEGER',
            'TOT_DURATION': 'INTEGER',
            'NRS_DURATION': 'INTEGER',
            'DUP_RATIO': 'REAL',
        }
        
        cmd = 'CREATE TABLE IF NOT EXISTS NRS('

        for k, v in table_elements.items():
            cmd += '{} {},'.format(k, v)

        cmd = cmd[:-1] + ')'

        self.cursor.execute(cmd)

        cmd = 'INSERT INTO NRS VALUES ("ROBOT", "GS", "R1", "01_ch1", 30.0, 25, 102058, 3038, 38.28)'

        self.cursor.execute(cmd)

        self.cursor.execute('SELECT * FROM NRS')
        rows = self.cursor.fetchall()

        for row in rows:
            print(row)

        df = pd.read_sql_query('SELECT * FROM NRS', self.connector)
        print(df)

        self.connector.commit()

    def add_column(self, info):
        cmd = 'ALTER TABLE NRS ADD COLUMN {} {}'.format(**info)
        self.cursor.execute(cmd)

    def select(self, query):
        pass

    def update(self, query):
        pass

    def insert(self, query):
        pass

    def delete(self, query):
        pass