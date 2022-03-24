import os
import pandas as pd



p = './logs/mobilenetv3_large_100-2-mini_fold_stage_0-offline-1/hem_assets/hem-softmax_diff_small-offline-agg.csv'
df = pd.read_csv(p)

print(df.head())
dv = df.values

cnt = 0
for v in dv:
    print(v)
    cnt += 1
    
    if cnt > 3:
        break