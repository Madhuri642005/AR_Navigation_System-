# ==========================================================
# AI Vision Guide v41 — Object Names + Distances + Navigation
# + On-Screen Captions + Auto Camera Detection
# ==========================================================
import os, time, threading, queue, cv2, numpy as np, pyttsx3, sounddevice as sd
from ultralytics import YOLO
from collections import deque, defaultdict
import textwrap

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8s.pt"   # download from Ultralytics if not present
CONF_OBJ, CONF_PERSON = 0.65, 0.82
VOICE_RATE = 172
FRAME_SKIP = 2

# STOP tuned for ~0.4–0.6 m range
STOP_RATIO = 0.27
BOTTOM_FORCE_STOP_RATIO = 0.14
BOTTOM_NEAR_BAND = 0.25

DIST_MIN, DIST_MAX = 0.4, 6.0
CLEAR_FRAMES = 8
REPEAT_CLEAR_SEC = 3.5
FAR_REPEAT_SEC = 4.0
NAV_REPEAT_SEC = 2.5

CLASS_WEIGHT = {"person":1.2,"wall":1.1,"door":1.0,"plants":0.65,"obstacle":1.0,"car":1.0,"chair":0.9,"table":0.9}
MAX_CLASS_CONTRIB_PER_ZONE = 1.4
SAFE_CLASSES = {"person","chair","table","car","bus","truck","door","bottle","cup","bench","sofa","bed","bicycle","motorcycle","plant","dog","cat","potted plant"}

# ---------------- CAMERA AUTO-DETECT ----------------
def get_camera():
    print("🎥 Searching for available camera...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Using camera index {i}")
            return cap
        cap.release()
    raise SystemExit("❌ No working camera found. Check your webcam or permissions.")

# ---------------- FACE DETECTOR ----------------
FACE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
FACE_CASCADE = cv2.CascadeClassifier(FACE_PATH)
FACE_OK = not FACE_CASCADE.empty()

def has_face(gray, box):
    if not FACE_OK: return True
    x1, y1, x2, y2 = box
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0: return False
    faces = FACE_CASCADE.detectMultiScale(roi, 1.1, 5, minSize=(24,24))
    return len(faces) > 0

# ---------------- SOFT BEEP ----------------
def beep(freq=900, dur=0.2):
    try:
        t = np.linspace(0, dur, int(44100 * dur), False)
        tone = 0.25 * np.sin(freq * 2 * np.pi * t)
        sd.play(tone, 44100)
        sd.wait()
    except Exception:
        pass

# ---------------- OVERLAY ----------------
class MessageOverlay:
    def __init__(self, hold_sec=2.6, fade_sec=0.25):
        self.queue = deque()
        self.current = None
        self.hold, self.fade = hold_sec, fade_sec
        self.pad, self.font_scale, self.font_thick = 14, 0.7, 2

    def show(self, text):
        text = " ".join(text.strip().split())
        if text: self.queue.append(text)

    def _rounded(self, img, x1, y1, x2, y2, color=(30,30,30), alpha=0.6, r=10):
        overlay = img.copy()
        cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, -1)
        cv2.circle(overlay, (x1+r, y1+r), r, color, -1)
        cv2.circle(overlay, (x2-r, y1+r), r, color, -1)
        cv2.circle(overlay, (x1+r, y2-r), r, color, -1)
        cv2.circle(overlay, (x2-r, y2-r), r, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    def draw(self, frame):
        H, W = frame.shape[:2]
        now = time.time()
        if self.current is None and self.queue:
            self.current = (self.queue.popleft(), now)
        if self.current is None: return frame
        text, t0 = self.current
        elapsed = now - t0
        if elapsed >= self.hold + self.fade:
            self.current = None
            return frame
        max_chars = max(8, W // 22)
        lines = textwrap.wrap(text, width=max_chars)
        sizes = [cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thick)[0] for l in lines]
        text_w = max(s[0] for s in sizes); text_h = sum(s[1] for s in sizes) + 10 * (len(lines)-1)
        box_w, box_h = text_w + 40, text_h + 30
        x1 = (W - box_w)//2; y2 = H - int(0.05*H); y1 = y2 - box_h; x2 = x1 + box_w
        self._rounded(frame, x1, y1, x2, y2)
        y = y1 + 20
        for (line, (lw, lh)) in zip(lines, sizes):
            cv2.putText(frame, line, (x1 + 20, y + lh), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, (255,255,255), self.font_thick, cv2.LINE_AA)
            y += lh + 10
        return frame

# ---------------- VOICE ----------------
class Voice(threading.Thread):
    def __init__(self, overlay=None, rate=VOICE_RATE):
        super().__init__(daemon=True)
        self.q = queue.Queue(); self.last_text = ""; self.last_time = 0
        self.overlay = overlay
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", rate)
        except Exception:
            self.engine = None
        self.start()

    def run(self):
        while True:
            text = self.q.get()
            if text is None: break
            print("🔊", text)
            if self.engine:
                try:
                    self.engine.stop()
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception: pass
            self.q.task_done()

    def speak(self, text, urgent=False):
        now = time.time()
        if not urgent and text == self.last_text and (now - self.last_time) < 1.0:
            return
        while not self.q.empty():
            try: self.q.get_nowait(); self.q.task_done()
            except: break
        self.q.put(text)
        self.last_text, self.last_time = text, now
        if self.overlay: self.overlay.show(text)

# ---------------- UTILITIES ----------------
def area_ratio(x1,y1,x2,y2,W,H): return max(0,(x2-x1))*max(0,(y2-y1))/float(W*H)
def direction_from_center(cx,cy,W,H):
    horiz = "center"
    if cx < W/3: horiz = "left"
    elif cx > 2*W/3: horiz = "right"
    return horiz
def zone_for_box(x1,y1,x2,y2,W):
    cx = (x1+x2)//2
    if cx < W/3: return "L"
    if cx > 2*W/3: return "R"
    return "C"
def nav_from_zones(z):
    L,C,R=z["L"],z["C"],z["R"]
    bl,bc,br = (L>=1.0),(C>=1.0),(R>=1.0)
    if not bl and not bc and not br: return "Path is clear. Move straight."
    if bc and not bl and R>L: return "Obstacle ahead. Move left."
    if bc and not br and L>R: return "Obstacle ahead. Move right."
    if bl and not bc and not br: return "Obstacle on left. Move right."
    if br and not bc and not bl: return "Obstacle on right. Move left."
    return "Obstacle ahead. Move carefully."

def estimate_distance(area_r): return float(np.clip(0.55/np.sqrt(max(area_r,1e-6)), 0.4, 6.0))
def distance_phrase(d):
    if d < 1.0: return "less than 1 meter"
    n = int(round(d)); return f"{n} meters" if n<=4 else f"about {n} meters"

# ---------------- MAIN PROGRAM ----------------
print("🔍 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ YOLO loaded")

cap = get_camera()
overlay = MessageOverlay()
voice = Voice(overlay=overlay)

print("\n👁️ AI Vision Guide started\nPress 'q' to quit.\n")

frame_idx = 0; last_nav = ""

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        frame = overlay.draw(frame)
        cv2.imshow("AI Vision Guide", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    H,W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    zone = {"L":0,"C":0,"R":0}
    results = model(frame, conf=CONF_OBJ, verbose=False)

    stop = False
    for r in results:
        if getattr(r, "boxes", None) is None: continue
        for b in r.boxes:
            cls = int(b.cls[0]); label = model.names[cls]
            conf = float(b.conf[0])
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            bw,bh = x2-x1, y2-y1
            if bw<40 or bh<40: continue
            if label=="person" and conf<CONF_PERSON: continue
            if label!="person" and conf<CONF_OBJ: continue
            ar = area_ratio(x1,y1,x2,y2,W,H)
            z = zone_for_box(x1,y1,x2,y2,W)
            d = estimate_distance(ar)
            zone[z] += min(1.0, ar*5.0)
            if ar >= STOP_RATIO:
                stop=True
                msg = f"Stop please! {label} very close, {distance_phrase(d)}."
                voice.speak(msg, urgent=True)

    nav = nav_from_zones(zone)
    if not stop and nav != last_nav:
        voice.speak(nav)
        last_nav = nav

    frame = overlay.draw(frame)
    cv2.imshow("AI Vision Guide", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

voice.speak("System stopped safely.")
print("\n🛑 System stopped safely.")
cap.release()
cv2.destroyAllWindows()
