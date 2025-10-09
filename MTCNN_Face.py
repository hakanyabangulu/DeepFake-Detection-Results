import os
import cv2
from tqdm import tqdm
from facenet_pytorch import MTCNN


VIDEO_FOLDER = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\Videolar"
FRAME_FOLDER = r"C:\Users\hakan\Desktop\School\4.th Grade\1.st Term\Bitirme Projesi\DeepFake Detection\MTCNN_Frames"

FRAMES_PER_VIDEO = 5
FRAME_SAMPLE_RATE = 5
DEVICE =  'cpu'

# ----- MTCNN YÜZ ALGILAMA -----
mtcnn = MTCNN(keep_all=True, device=DEVICE)

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
                if not ret or frame is None or frame.size == 0:
                    break

                if current % FRAME_SAMPLE_RATE == 0:
                    # MTCNN için BGR -> RGB ve float32 normalize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    
                    try:
                        boxes, probs = mtcnn.detect(frame_rgb)
                    except Exception as e:
                        print(f"Detect error ({video}): {e}")
                        continue

                    if boxes is not None and len(boxes) > 0:
                        best_idx = probs.argmax()
                        x1, y1, x2, y2 = map(int, boxes[best_idx])
                        x1, y1 = max(0, x1), max(0, y1)
                        w, h = x2 - x1, y2 - y1
                        face_crop = frame[y1:y1+h, x1:x1+w]
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
