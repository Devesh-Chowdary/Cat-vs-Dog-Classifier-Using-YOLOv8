# Cat-vs-Dog-Classifier-Using-YOLOv8

# 🐱🐶 YOLOv8 Cat vs Dog Classifier

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

### ✅ Step 1: Install YOLOv8
Install the required YOLOv8 package from Ultralytics.
```python
!pip install ultralytics
