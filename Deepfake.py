import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, recall_score, f1_score, precision_score
import numpy as np




# ----- PATHS -----
VIDEO_FOLDER = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\Videolar"
FRAME_FOLDER = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\Frames\Yunet_Frames"
TEST_VIDEO_FOLDER = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\TestVideolar"
MODEL_PATH = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\Models\cnn_classifier.pth"
FACE_MODEL_PATH = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\Models\face_detection_yunet_2023mar.onnx"

BATCH = 32
EPOCHS = 15
LR = 1e-3
FRAMES_PER_VIDEO = 8
THRESHOLD = 0.5
FRAME_SAMPLE_RATE = 5
CONFIDENCE_THRESHOLD = 0.94

# ----- YÜZ ALGILAMA -----
if not os.path.exists(FACE_MODEL_PATH):
    raise ValueError(f"YuNet model dosyası bulunamadı: {FACE_MODEL_PATH}")
face_detector = cv2.FaceDetectorYN.create(FACE_MODEL_PATH, "", (320, 320))

# ----- FRAME EXTRACTION -----
def extract_frames(video_files=None, video_folder=VIDEO_FOLDER, output_folder=FRAME_FOLDER, log_callback=None):
    os.makedirs(output_folder, exist_ok=True)
    for label in ["real", "fake"]:
        input_dir = os.path.join(video_folder, label)
        output_dir = os.path.join(output_folder, label)
        os.makedirs(output_dir, exist_ok=True)
        videos = video_files if video_files else [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
        for video in tqdm(videos, desc=f"Processing {label}"):
            video_path = video if video_files else os.path.join(input_dir, video)
            cap = cv2.VideoCapture(video_path)
            current, saved = 0, 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current % FRAME_SAMPLE_RATE == 0:
                    img_H, img_W = frame.shape[:2]
                    face_detector.setInputSize((img_W, img_H))
                    _, detections = face_detector.detect(frame)
                    if detections is not None and len(detections) > 0:
                        best_face, max_confidence = None, 0
                        for detection in detections:
                            confidence = detection[-1]
                            if confidence > max_confidence and confidence >= CONFIDENCE_THRESHOLD:
                                max_confidence, best_face = confidence, detection
                        if best_face is not None:
                            x, y, w, h = map(int, best_face[:4])
                            x, y = max(0, x), max(0, y)
                            w, h = min(w, img_W - x), min(h, img_H - y)
                            if w > 0 and h > 0:
                                face_crop = frame[y:y+h, x:x+w]
                                if face_crop.size > 0:
                                    face_crop = cv2.resize(face_crop, (256, 256))
                                    frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video))[0]}_{current}_face.jpg")
                                    cv2.imwrite(frame_path, face_crop)
                                    saved += 1
                                    if saved >= FRAMES_PER_VIDEO:
                                        break
                current += 1
            cap.release()
            msg = f"{video} -> {saved} kare kaydedildi"
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

# ----- DATASET -----
class DeepFakeDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data, self.labels, self.transform = [], [], transform
        for label in ["real", "fake"]:
            folder = os.path.join(root, label)
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
            self.data.extend(files)
            self.labels.extend([0 if label == "real" else 1] * len(files))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ----- MODEL -----
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ----- TRAIN -----

def train_model(log_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomResizedCrop(256, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.02, 0.1))
    ])
    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    full_dataset = DeepFakeDataset(FRAME_FOLDER, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc, best_val_loss = 0.0, float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels, all_val_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total

        scheduler.step()
        print(f"Epoch {epoch+1}: TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✔ Yeni en iyi model kaydedildi (ValAcc={val_acc:.4f}, ValLoss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ Early stopping.")
                break

    # ---- Validation sonu: en iyi threshold bulma ----
    
    prec, rec, thr = precision_recall_curve(all_val_labels, all_val_probs)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.nanargmax(f1[:-1])
    best_thr = thr[best_idx]
    print(f"\n🎯 Validation tabanlı en iyi eşik (THRESHOLD): {best_thr:.3f} | F1={f1[best_idx]:.4f}")
    return model, best_thr


# ----- TEST -----
def test_model(threshold=0.5, log_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.ToPILImage(), T.Resize((256, 256)), T.ToTensor()])
    criterion = nn.CrossEntropyLoss()

    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    total_correct, total_videos, running_loss = 0, 0, 0.0
    all_preds, all_labels = [], []

    for label in ["real", "fake"]:
        folder = os.path.join(TEST_VIDEO_FOLDER, f"test_{label}")
        if not os.path.exists(folder):
            print(f"Test klasörü bulunamadı: {folder}")
            continue
        ground_truth = 0 if label == "real" else 1
        videos = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4")]

        for video_path in tqdm(videos, desc=f"Testing {label.capitalize()}"):
            cap = cv2.VideoCapture(video_path)
            frames, current = [], 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current % FRAME_SAMPLE_RATE == 0:
                    img_H, img_W = frame.shape[:2]
                    face_detector.setInputSize((img_W, img_H))
                    _, detections = face_detector.detect(frame)
                    if detections is not None:
                        best_face = None
                        max_conf = 0
                        for det in detections:
                            conf = det[-1]
                            if conf > max_conf and conf >= CONFIDENCE_THRESHOLD:
                                best_face = det
                                max_conf = conf
                        if best_face is not None:
                            x, y, w, h = map(int, best_face[:4])
                            x, y = max(0, x), max(0, y)
                            w, h = min(w, img_W - x), min(h, img_H - y)
                            face_crop = frame[y:y+h, x:x+w]
                            if face_crop.size > 0:
                                face_crop = cv2.cvtColor(cv2.resize(face_crop, (256, 256)), cv2.COLOR_BGR2RGB)
                                frames.append(transform(face_crop))
                                if len(frames) >= FRAMES_PER_VIDEO:
                                    break
                current += 1
            cap.release()

            if not frames:
                continue

            frames_tensor = torch.stack(frames).to(device)
            labels_tensor = torch.tensor([ground_truth] * len(frames), dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(frames_tensor)
                loss = criterion(outputs, labels_tensor)
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                confidence = probs.median().item()  # 🔥 MEAN yerine MEDIAN
                pred_label = 1 if confidence > threshold else 0
                correct = int(pred_label == ground_truth)

            total_correct += correct
            total_videos += 1
            all_preds.append(pred_label)
            all_labels.append(ground_truth)

    
    acc = total_correct / total_videos if total_videos else 0
    loss = running_loss / total_videos if total_videos else 0
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"\nTest Results -> Accuracy: {acc:.4f}, Average Loss: {loss:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return acc, prec, rec, f1


# ----- MAIN -----
if __name__ == "__main__":
    extract_frames()
    model, best_thr = train_model()
    print(f"\n🚀 Test aşamasında {best_thr:.3f} eşiği kullanılacak...")
    test_model(threshold=best_thr)

    
    
    
