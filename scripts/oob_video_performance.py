# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:47:51 2020

@author: danielvandencorput
"""
import os
import pandas as pd
from scipy.signal import medfilt

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path
def convert_to_fps(time_list,fps):
    out = []
    for time in time_list:
        for t in time:
            out.append(sum(minute*int(second) for minute,second in zip([60,1], t.split(':')))*fps)
    return out


### Nth frame to predict and threshold
frames = 75
THRESHOLD = 0.8
### Set up paths and files
output_folder = os.path.join(get_parent_dir(1),'Output')
output_file = os.path.join(output_folder,'incep_6min_{}frames.csv'.format(frames))
### Read predictions
output = pd.read_csv(output_file)
max_frame = output['frame'].max()
### Read ground truth file
gt_file = os.path.join(output_folder,'6min_gt.csv')
gt = pd.read_csv(gt_file,sep=';')
### Convert frames to fps
df_out = []
s_times = convert_to_fps(gt['oob_start'].str.split(';'),25)
e_times = convert_to_fps(gt['oob_end'].str.split(';'),25)
times = [range(start, stop+1) for start, stop in zip(s_times, e_times)]
### Convert ground truth to congruent length with predictions
for i in range(0,max_frame,frames):
    if any([i in my_range for my_range in times]):
        d = {
            'frame':i,
            'label':1}
        df_out.append(d)
    else:
        d = {
            'frame':i,
            'label':0}
        df_out.append(d)
        
y_true = pd.DataFrame(df_out)
y_true = list(y_true['label'])

### Convert labels to numeric
output['label'] = [1 if output.iloc[e]['conf_out'] >= THRESHOLD else 0 for e in range(len(output))]
y_pred = list(output['label'])

### Get accuracy score
def accuracy(y_true, y_pred):
    assert len(y_true)==len(y_pred), "Mismatching lengths"
    return sum([y_true[i]==y_pred[i] for i in range(len(y_true)) ]) / len(y_true)

### Median filter
f1 = medfilt(y_pred,13)

print(accuracy(y_true, f1))