# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import imageio
import sys
import os
import time
import cv2

import pandas as pd
import numpy as np
import tensorflow as tf

from skimage import img_as_ubyte
from scipy.signal import medfilt
from datetime import datetime

tf.compat.v1.disable_eager_execution()

###############################################################################
####################### PREDICT AND WRITE BLURRED VIDEO #######################
###############################################################################

# This script needs a video in the ./Data/videos/ folder. It will create a
# prediction for every 30th frame in the video and subsequently use those
# predictions to blur the out of body moments. 
# The blurred video will be saved to the ./Output/videos/ folder.

###############################################################################

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

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
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


if __name__ == "__main__":
  file_dir = os.path.join(get_parent_dir(),"tf_files","test")          
  model_file = os.path.join(get_parent_dir(),"tf_files","retrained_graph.pb")
  label_file = os.path.join(get_parent_dir(),"tf_files","retrained_labels.txt")
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  parser.add_argument("--video", help="name of input video")
  parser.add_argument("--output_file", help="name of output file",default='conf_scores.csv')
  args = parser.parse_args()


  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  if args.video:
      input_video = args.video
  if args.output_file:
      out_file = args.output_file

video_path = os.path.join(get_parent_dir(1),'Data','videos')

video = os.path.join(video_path,#YOUR VIDEO HERE)
output_folder = os.path.join(get_parent_dir(1),'Output','predictions')
output_file = os.path.join(output_folder, 'conf_scores_{}.csv'.format(video.split('/')[-1].split('.')[0]))
cap = cv2.VideoCapture(video)
df = pd.DataFrame(columns=['frame','conf_out','conf_in'])
currentframe = 0

graph = load_graph(model_file)

with tf.compat.v1.Session(graph=graph) as sess:
    
    while cap.isOpened():

    # Read frame
        ret,frame = cap.read()
    
        if not ret:
            break
        if frame is None:
            continue
        if currentframe % 30 == 0:

            # Use smoothed frame for prediction
            name = str(currentframe)
            # t = read_tensor_from_image_file(cv2.boxFilter(frame,0,(15,15)),\
            t = read_tensor_from_image_file(frame,
                                            input_height=299,
                                            input_width=299,
                                            input_mean=input_mean,
                                            input_std=input_std)
        
            input_name = "import/" + "Mul"
            output_name = "import/" + output_layer
            input_operation = graph.get_operation_by_name(input_name);
            output_operation = graph.get_operation_by_name(output_name);
            # Use trained model to predict results
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
            end=time.time()
            results = np.squeeze(results)
            out = []
            top_k = results.argsort()[-5:][::-1]
            labels = load_labels(label_file)
            [out.append({'frame':name,'conf_out':results[1],'conf_in':results[0]})]                 
            # Append to df with filename and confidence scores per frame
            df = df.append(out)    
            print('Predicted frame', currentframe)
            currentframe += 1            
        else:
            currentframe += 1
            continue
        if not ret:
            break
        
    
df.to_csv(output_file,header=True,index=False)
cap.release()
cv2.destroyAllWindows() 
print('[INFO] Predictions are made. Now blurring video based on predictions.')

csv = output_file
df_in = pd.read_csv(csv)
write_to = os.path.join(get_parent_dir(1),'Output','videos','blurred_video_{}.mp4'.format(video.split('/')[-1].split('.')[0]))

THRESHOLD = 0.7
labels = [1 if df_in.iloc[e]['conf_out']>THRESHOLD else 0 for e in range(len(df_in))]

f1 = medfilt(labels, 7)
df_in['labels'] = f1

s_time = datetime.now()

cap = cv2.VideoCapture(video)
fps = cap.get(5)
currentframe = 0

writer = imageio.get_writer(write_to,format='mp4',mode='I',fps=fps)

while cap.isOpened():
    try:
        ret,frame = cap.read()

        if not ret:
            break

        if frame is None:
            continue

        idx = np.argmin(np.abs(df_in['frame']-(currentframe+10)))

        if df_in['labels'].loc[idx] == 1:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(img_as_ubyte(img))
            
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(img_as_ubyte(img))

        currentframe += 1

        if currentframe % 50 == 0:
            print('Written video frame',currentframe)

    except ValueError:
        continue

writer.close()
e_time = datetime.now()
print('Elapsed time:',e_time-s_time)
print('[INFO] Done.')
