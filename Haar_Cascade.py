import os
import cv2
from tqdm import tqdm

VIDEO_FOLDER = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\Videolar"
FRAME_FOLDER = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\Haar_Frames"
MODEL_PATH = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\Models\haarcascade_frontalface_default.xml"

BATCH = 32
EPOCHS = 10
LR = 1e-3
FRAMES_PER_VIDEO = 5
FRAME_SAMPLE_RATE = 5

# ----- HAAR CASCADE YÜZ ALGILAMA -----
if not os.path.exists(MODEL_PATH):
    raise ValueError(f"Haar Cascade model dosyası bulunamadı: {MODEL_PATH}")

face_detector = cv2.CascadeClassifier(MODEL_PATH)

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
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

                    if len(faces) > 0:
                        # En büyük yüzü al
                        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                        x, y = max(0, x), max(0, y)
                        face_crop = frame[y:y+h, x:x+w]
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
                
if __name__ == "__main__":
    extract_frames()