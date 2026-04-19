# 🚦 Indian Traffic Sign Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-orange?style=flat&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=flat&logo=flask)

A deep learning web app that recognizes **59 Indian Traffic Signs** using a Custom CNN (PyTorch) + Flask.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision flask numpy pillow scikit-learn opencv-python

# 2. Add dataset → dataset/Train/0/, 1/, 2/ ...
#    Download: https://www.kaggle.com/datasets/neelpratiksha/indian-traffic-sign-dataset

# 3. Train the model
python train.py

# 4. Run the website
python app.py
# Open → http://localhost:5000
```

---

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| Deep Learning | PyTorch (Custom 3-Block CNN) |
| Web App | Flask |
| Image Processing | OpenCV |
| Frontend | HTML5, CSS3, JavaScript |
| Dataset | Indian Traffic Sign — Kaggle (59 classes) |

---

## 🌐 How to Use

1. Open `http://localhost:5000`
2. Drag & Drop OR upload a traffic sign image
3. Get **Top-5 predictions** with confidence scores instantly


