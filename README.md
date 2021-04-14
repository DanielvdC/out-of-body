# Out of body
Project to detect whether laparoscopic video frames are either inside or outside the patient's body.
 
This project includes a pre-trained model, which is the Inception_v3, trained on about 30.000 images. This model is stored under ./tf_files/retrained_graph.pb.
If you wish to retrain with your own dataset, use the ./scripts/retrain.py file to do so.
 
With these scripts you can create in and out patient detections for new videos, plot these predictions and use them in order to blur the out of body frames.

In order to do so, follow these steps:
1. Clone the repository into your environment
2. Install necessary packages with:
<br> pip install -r requirements.txt
3. Either retrain with your own images by using the ./scripts/retrain.py script
4. Or add your videos to the ./Data/videos/ folder and run the ./scripts/video_predict_blur.py script
5. The blurred video will be saved to ./Output/videos/


# Author
Daniël van den Corput
danielvdcorput@gmail.com

# Purpose
This project was created for Incision, see https://www.incision.care/
