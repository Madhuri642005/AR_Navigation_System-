# ==========================================================
# AI Vision Guide v22 — Instant Object + Name + Navigation
# ==========================================================
# ✅ Instantly speaks detected object names & direction
# ✅ Face-aware person detection (optional)
# ✅ Real-time navigation feedback (clear path / obstacles)
# ✅ No voice lag, no overlap, CPU smooth
# ==========================================================

from ultralytics import YOLO
import cv2, os, time, threading, pyttsx3, queue, numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.6
PERSON_CONF_THRESH = 0.7
FRAME_WIDTH, FRAME_HEIGHT = 960, 540
IMG_SIZE = 640
VOICE_RATE = 170
VOICE_COOLDOWN = 1.0      # seconds before same object can be re-spoken
NAV_COOLDOWN = 2.0
FACE_REQUIRED = False      # ✅ Optional: set to True if you want face verification for "person"

# ---------------- FACE DETECTOR ----------------
FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)
def has_face_inside(gray, box):
    x1, y1, x2, y2 = box
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0: return False
    faces = FACE_CASCADE.detectMultiScale(roi, 1.1, 4, minSize=(25,25))
    return len(faces) > 0

# ---------------- SPEAKER (Instant, Non-blocking) ----------------
class Speaker:
    def __init__(self, rate=170):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        for v in self.engine.getProperty("voices"):
            if "en" in v.id.lower():
                self.engine.setProperty("voice", v.id)
                break
        self.q = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()
        self.last_spoken = {}

    def _worker(self):
        while True:
            key, text, cooldown = self.q.get()
            now = time.time()
            if key in self.last_spoken and now - self.last_spoken[key] < cooldown:
                continue
            with self.lock:
                self.engine.say(text)
                self.engine.runAndWait()
            self.last_spoken[key] = time.time()

    def speak(self, key, text, cooldown=1.0):
        self.q.put((key, text, cooldown))

speaker = Speaker(rate=VOICE_RATE)

# ---------------- HELPERS ----------------
def direction_from_center(cx, w):
    if cx < w/3: return "on your left"
    elif cx > 2*w/3: return "on your right"
    return "ahead"

def draw_box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, f"{label.upper()} {int(conf*100)}%", (x1, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ---------------- LOAD YOLO MODEL ----------------
print("🔍 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened(): raise SystemExit("❌ Camera not found")

print("🎯 AI Vision Guide v22 — Instant Detection + Navigation started.\n")

# ---------------- LOOP ----------------
last_nav_msg = ""
last_nav_time = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model(frame, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)
    zones = {"left": False, "center": False, "right": False}

    # ---------- Object Detection + Naming ----------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "person" and conf < PERSON_CONF_THRESH:
                continue
            if y2 - y1 < 40 or x2 - x1 < 40:
                continue

            if label == "person" and FACE_REQUIRED:
                if not has_face_inside(gray, (x1, y1, x2, y2)):
                    label = "person (?)"

            color = (0,255,0) if "person" in label else (255,180,0)
            draw_box(frame, x1, y1, x2, y2, label, conf, color)

            cx = (x1+x2)//2
            dirn = direction_from_center(cx, w)
            phrase = f"{label} {dirn}"
            speaker.speak(f"{label}:{dirn}", phrase, VOICE_COOLDOWN)

            # zones for navigation
            if y2 > h * 0.55:
                if cx < w/3: zones["left"] = True
                elif cx > 2*w/3: zones["right"] = True
                else: zones["center"] = True

    # ---------- Navigation Feedback ----------
    msg = ""
    if not any(zones.values()):
        msg = "Path is clear, move ahead"
    elif zones["center"]:
        if not zones["left"]:
            msg = "Obstacle ahead, move slightly left"
        elif not zones["right"]:
            msg = "Obstacle ahead, move slightly right"
        else:
            msg = "Obstacle blocking your way, stop"
    elif zones["left"]:
        msg = "Obstacle on your left, move right"
    elif zones["right"]:
        msg = "Obstacle on your right, move left"

    now = time.time()
    if msg and (msg != last_nav_msg or now - last_nav_time > NAV_COOLDOWN):
        speaker.speak("nav", msg, NAV_COOLDOWN)
        last_nav_msg = msg
        last_nav_time = now

    cv2.putText(frame, msg, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.imshow("👁️ AI Vision Guide v22 — Instant Object + Navigation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Stopped safely.")
