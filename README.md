# Cat 🐱 -vs-Dog 🐶 -Classifier-Using-YOLOv8


This project trains a **YOLOv8 object detection model** to classify and localize **cats and dogs** in images. It is designed to run entirely on **Google Colab with GPU support** using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework.

---

## 📌 Features
- ✅ Trains YOLOv8 nano (`yolov8n.pt`) from scratch on a custom Cat vs Dog dataset.
- ⚡ GPU-enabled training on Google Colab.
- 📦 Easy to test your own images.
- 🎯 High-speed inference with bounding boxes.

---

## 🔧 Requirements
- Python ≥ 3.8
- Google Colab (GPU enabled)
- Required libraries: `ultralytics`, `Pillow`

---

## 🚀 Step-by-Step Instructions

### **Step 1: Enable GPU in Google Colab**
Before running the code:
1. Go to **Runtime > Change runtime type**.
2. Set **Hardware accelerator = GPU**.
3. Save and restart the runtime.

---

### **Step 2: Install YOLOv8**
Install the required YOLOv8 package from Ultralytics.
```python
!pip install ultralytics
```

---

### **Step 3: Upload and Extract Dataset**
Upload the custom YOLO-formatted dataset zip and extract it:

```python
from google.colab import files
uploaded = files.upload()   # Upload CAT VS DOG.v1i.yolov8.zip

import zipfile
with zipfile.ZipFile("CAT VS DOG.v1i.yolov8.zip", 'r') as zip_ref:
    zip_ref.extractall("/content")
!mv "/content/CAT VS DOG.v1i.yolov8" "/content/cat_dog_data"
```

Once extracted, your folder should look like:
```python
/content/cat_dog_data/
├── images/
├── labels/
└── data.yaml
```
---

### **Step 4: Train the YOLOv8 Model on GPU**
Train the model from scratch using the YOLOv8 Nano weights:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

model.train(
    data="/content/cat_dog_data/data.yaml",  
    epochs=30,          
    imgsz=640,          
    device=0            
)
```
After training, weights will be saved at:
/content/runs/detect/train/weights/best.pt

---

### **Step 5: Test the Model on a New Image**
Upload any test image:
```python
from google.colab import files
uploaded = files.upload()  # Upload an image
```
Then run inference:
```python
from ultralytics import YOLO

model = YOLO("/content/runs/detect/train/weights/best.pt")  
results = model.predict(source="test1.jpg", save=True)       
```

Predicted images will be saved in a folder like:
/content/runs/detect/predict/

### **Step 6: Display the Result**
Show the predicted output using PIL:
```python
from PIL import Image
Image.open('/content/runs/detect/predict/test1.jpg')
```
This will display the image with bounding boxes around detected cats and dogs.

### **Sample Output**




