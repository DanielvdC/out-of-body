import os
import pandas as pd
import numpy as np
import cv2
import imageio
import time

from skimage import img_as_ubyte
from scipy.signal import medfilt


def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

folder = os.path.join(get_parent_dir(1))
video = os.path.join(folder,'Data','videos','MK_full_video.mp4')
csv = os.path.join(folder,'Output','MK_full_output.csv')
write_to = os.path.join(folder,'Output','videos','MK_full_video_blurred.mp4')

df_in = pd.read_csv(csv,sep=';')

THRESHOLD = 0.9
labels = [1 if df_in.iloc[e]['conf_out']>THRESHOLD else 0 for e in range(len(df_in))]

f1 = medfilt(labels, 7)
df_in['labels'] = f1 

start = time.time()

cap = cv2.VideoCapture(video)
fps = cap.get(5)

writer = imageio.get_writer(write_to,format='mp4',mode='I',fps=fps)


font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText    = (50,50)
fontScale              = 0.5
fontColor              = (0,255,0)
lineType               = cv2.LINE_4

while cap.isOpened():
    try: 
        ret,frame = cap.read()
        currentframe = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            break    
        if frame is None:
            continue
        
        idx = min(range(len(df_in['frame'])), key=lambda i: abs(df_in['frame'][i]-currentframe))
        
        if df_in['labels'].loc[idx] == 1:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.blur(img,(200,200))
            cv2.putText(img,'[out body]', 
                        topLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
            writer.append_data(img_as_ubyte(img))
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(img,'[in body]', 
                        topLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
            writer.append_data(img_as_ubyte(img))
            
        if currentframe % 25 == 0:
            print(currentframe)
                
    except ValueError:
        continue
        
        
writer.close()
end = time.time()
print('Elapsed time: {}s'.format(end-start)) 

