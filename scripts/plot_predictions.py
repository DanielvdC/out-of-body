# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:47:51 2021

@author: danielvandencorput
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import medfilt

###############################################################################
############################### PLOT PREDICTIONS ##############################
###############################################################################

# This script takes the generated predictions and plots them over a timeline.

###############################################################################

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

### Input paths
folder = os.path.join(get_parent_dir(1))
video = os.path.join(folder,'Data',#YOUR VIDEO HERE)
csv_file = os.path.join(folder,'Output',#YOUR .csv PREDICTIONS HERE)
### FPS of video
FPS = 25
### Out of body threshold
THRESHOLD = 0.9

### Read csv
df = pd.read_csv(csv_file,sep=';')

### Process dataframe
plot = pd.DataFrame(columns=['time','label'])
time = [frame/FPS/60 for frame in df['frame']]
label = [1 if df.iloc[e]['conf_out'] > THRESHOLD else 0 for e in range(len(df))]
f1 = medfilt(label,7)
#f1 = medfilt(label[0:1250],17)
#f2 = medfilt(label[1250:],9)
#f1 = np.append(f1, f2)

plot['time'], plot['label'] = time, f1

### Set style
sns.set(font_scale = 1.2)
sns.set_style(style = 'white')
### Plot relational line plot
g = sns.relplot(x = 'time',
                y = 'label',
                data = plot,
                kind = 'line',
                palette = ['#5dbcd2'])
g.fig.set_size_inches(9.5, 2)

### Adjust ticks and labels
g.set(yticks=[0,1])
g.set(yticklabels=['in body','out body'])
g.set(ylabel=None)
g.set(xlabel='time (mins)')

plt.show()













