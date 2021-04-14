#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:58:28 2021

@author: danielvdcorput
"""
import os
import pandas as pd
import seaborn as sns

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

output_folder = os.path.join(get_parent_dir(1),'Output')
gt_file = os.path.join(output_folder,'6min_gt.csv')
gt = pd.read_csv(gt_file,sep=';')

predictions = os.path.join(output_folder,'incep_6min_50frames.csv')
predictions = pd.read_csv(predictions)
max_frame = predictions['frame'].max()
predictions['time'] = [frame/25/60 for frame in predictions['frame']]

THRESHOLD = 0.9
predictions['label'] = [1 if predictions.iloc[e]['conf_out'] > THRESHOLD
                        else 0 for e in range(len(predictions))]
predictions['label'] = medfilt(predictions['label'],23)

### Convert frames to fps
df_out = []
s_times = convert_to_fps(gt['oob_start'].str.split(';'),25)
e_times = convert_to_fps(gt['oob_end'].str.split(';'),25)
times = [range(start, stop+1) for start, stop in zip(s_times, e_times)]
### Convert ground truth to congruent length with predictions
for i in range(0,max_frame,50):
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
y_true['time'] = [frame/25/60 for frame in y_true['frame']]


sns.set(font_scale = 1.2)
sns.set_style(style = 'white')
q = sns.relplot(x = 'time',
                y = 'label',
                data = y_true,
                kind = 'line',
                palette = ['#5dbcd2'],
                legend=False)
q.fig.set_size_inches(9.5, 2)
q.fig.suptitle('Actual labels',y=1.2)
q.set(yticks=[0,1])
q.set(yticklabels=['in body','out body'])
q.set(ylabel=None)
q.set(xlabel='time (mins)')
g = sns.relplot(x = 'time',
                y = 'label',
                data = predictions,
                kind = 'line',
                palette = ['#5dbcd2'],
                legend=False)
g.fig.set_size_inches(9.5, 2)
g.fig.suptitle('Predicted labels',y=1.2)
g.set(yticks=[0,1])
g.set(yticklabels=['in body','out body'])
g.set(ylabel=None)
g.set(xlabel='time (mins)')











