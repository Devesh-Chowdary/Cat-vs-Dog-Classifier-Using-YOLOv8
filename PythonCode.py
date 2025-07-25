
!pip install ultralytics

from google.colab import files
uploaded = files.upload()   # Upload your dataset ZIP here

import zipfile
with zipfile.ZipFile("CAT VS DOG.v1i.yolov8.zip", 'r') as zip_ref:
    zip_ref.extractall("/content")

!mv "/content/CAT VS DOG.v1i.yolov8" "/content/cat_dog_data"

import os
os.listdir("/content/cat_dog_data")

from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data="/content/cat_dog_data/data.yaml",  
    epochs=30,          
    imgsz=640,          
    device=0            
)

from google.colab import files
uploaded = files.upload() 

from ultralytics import YOLO
model = YOLO("/content/runs/detect/train/weights/best.pt")
results = model.predict(source="test1.jpg", save=True)

import os
os.listdir('/content/runs/detect/predict3')

from PIL import Image
Image.open('/content/runs/detect/predict3/test4.jpg')

