import os
import uuid
import numpy as np
import torch
import torch.nn as nn
import cv2
from flask import Flask, render_template, request, jsonify
from labels import classes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED  = {'png', 'jpg', 'jpeg', 'webp', 'ppm'}
IMG_SIZE = 32
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


with open("model/num_classes.txt") as f:
    NUM_CLASSES = int(f.read().strip())


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


print("Loading model...")
model = TrafficSignCNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("model/indian_ts_model.pth", map_location=DEVICE))
model.eval()
print("Model ready!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

def preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file. Use PNG, JPG or JPEG.'}), 400

    ext         = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath    = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    img = preprocess(filepath)
    with torch.no_grad():
        outputs = model(img)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    top5_idx = probs.argsort()[-5:][::-1]
    result = {
        'prediction': classes.get(int(top5_idx[0]), f'Class {top5_idx[0]}'),
        'confidence': round(float(probs[top5_idx[0]]) * 100, 2),
        'top5': [
            {'label': classes.get(int(i), f'Class {i}'),
             'confidence': round(float(probs[i]) * 100, 2)}
            for i in top5_idx
        ],
        'image_url': f'/static/uploads/{unique_name}'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)