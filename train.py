import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import pickle

DATA_PATH   = "dataset\Indian-Traffic Sign-Dataset\Images"
IMG_SIZE    = 32
BATCH_SIZE  = 64
EPOCHS      = 30
MODEL_SAVE  = "model/indian_ts_model.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class_folders = sorted(
    [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))],
    key=lambda x: int(x)
)
NUM_CLASSES = len(class_folders)
print(f"Detected {NUM_CLASSES} classes")


class TrafficSignDataset(Dataset):
    def __init__(self, data_path, class_folders, transform=None):
        self.images, self.labels = [], []
        self.transform = transform
        for class_id, folder in enumerate(class_folders):
            folder_path = os.path.join(data_path, folder)
            for img_file in os.listdir(folder_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                    continue
                img = cv2.imread(os.path.join(folder_path, img_file))
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images.append(img)
                self.labels.append(class_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # HWC → CHW
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

transform_train = transforms.Compose([
    transforms.RandomRotation(12),
    transforms.RandomAffine(0, translate=(0.1, 0.1), shear=10, scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

print("Loading dataset...")
full_dataset = TrafficSignDataset(DATA_PATH, class_folders, transform=transform_train)
val_size   = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
print(f"Train: {train_size} | Val: {val_size}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 3
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

model     = TrafficSignCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)


os.makedirs("model", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

best_val_acc = 0.0
patience_counter = 0
PATIENCE = 8

print("\nTraining started...")
for epoch in range(EPOCHS):

    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == lbls).sum().item()
        train_total   += lbls.size(0)

    
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outputs    = model(imgs)
            loss       = criterion(outputs, lbls)
            val_loss  += loss.item()
            preds      = outputs.argmax(dim=1)
            val_correct += (preds == lbls).sum().item()
            val_total   += lbls.size(0)

    train_acc = 100 * train_correct / train_total
    val_acc   = 100 * val_correct   / val_total
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
        f"Train Acc: {train_acc:.2f}%  "
        f"Val Acc: {val_acc:.2f}%  "
        f"Val Loss: {avg_val_loss:.4f}")


    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"Best model saved ({val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.2f}%")
print(f"Model saved → {MODEL_SAVE}")


with open("model/num_classes.txt", "w") as f:
    f.write(str(NUM_CLASSES))