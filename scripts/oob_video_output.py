# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:47:51 2020

@author: danielvandencorput
"""
### Libraries
import os
import time
import cv2

import pandas as pd
import numpy as np
import tensorflow as tf

from datetime import datetime

###############################################################################
############################# OUTPUT PREDICTIONS ##############################
###############################################################################

# This scripts loops over every frame of a provided video in order to generate
# out of body and in body predictions.
# Output file is a .csv file, used in the other scripts. Keep the name formats
# the same for compatibility with the other scripts.

###############################################################################

### Eager TF execution
start = time.time()
tf.compat.v1.disable_eager_execution()

### Functions
def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()
  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph

def read_tensor_from_image_file(file_name, input_height=338, input_width=338,
				input_mean=0, input_std=255):
  float_caster = tf.cast(file_name, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)
  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.compat.v2.io.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

### Paths and variables
label_file = os.path.join(get_parent_dir(1),"tf_files","retrained_labels.txt")
model_file = os.path.join(get_parent_dir(1),"tf_files","retrained_graph.pb")
graph = load_graph(model_file)
input_layer = "input"
output_layer = "final_result"
### Nth frame to predict
frames = 75

folder = os.path.join(get_parent_dir(1))
video = os.path.join(folder,'Data','videos','MK_full_video.mp4')
output_file = os.path.join(folder,'Output','MK_full_output.csv')

df = pd.DataFrame(columns=['frame','conf_in','conf_out'])

### Video
cap = cv2.VideoCapture(video)

### Predictions
with tf.compat.v1.Session(graph=graph) as sess:
    
    while cap.isOpened():
        ret,frame = cap.read()
        currentframe = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            break
        ### Resize option
        frame = cv2.resize(frame, (426,240))
        if currentframe % frames == 0:
            t = read_tensor_from_image_file(frame,
                                    input_height = 299, #224 / 299
                                    input_width = 299) #229 / 299
            input_name = "import/" + "Mul"
            output_name = "import/" + output_layer
            input_operation = graph.get_operation_by_name(input_name);
            output_operation = graph.get_operation_by_name(output_name);
            # Use trained model to predict results
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
            results = np.squeeze(results)

            out = [] 
            [out.append({'frame':currentframe,'conf_out':results[1],'conf_in':results[0]})]                 
            # Append to df with timestamp and confidence scores per frame
            df = df.append(out)   
            if currentframe % 25 == 0:
                print(currentframe)
                
df.to_csv(output_file,header=True,index=False)
cap.release()
cv2.destroyAllWindows()

end = time.time()
print('Elapsed time: {}s'.format(end-start))


        
        
        
        
        
        
