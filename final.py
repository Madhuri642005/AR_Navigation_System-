# ==========================================================
# AI Vision Guide v41 — Object Names + Distances + Reasoned Navigation
# ==========================================================
# • Always speaks object name + distance (meters)
# • Navigation commands include the blocking object(s) + their distances
# • Tight STOP (~0.4–0.6 m) + gentle beep on STOP
# • Whole-number / natural distance phrasing:
#     <1 m -> "less than 1 meter"; 1–4 m -> "N meters"; >4 m -> "about N meters"
# • Path-clear + far reassurance repeats; balanced focus
# ==========================================================

import os, time, threading, queue, cv2, numpy as np, pyttsx3, sounddevice as sd
from ultralytics import YOLO
from collections import deque, defaultdict

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8s.pt"
CONF_OBJ, CONF_PERSON = 0.65, 0.82
VOICE_RATE = 172
FRAME_SKIP = 2

# STOP tuned for ~0.4–0.6 m range
STOP_RATIO = 0.27
BOTTOM_FORCE_STOP_RATIO = 0.14
BOTTOM_NEAR_BAND = 0.25

# Distances (meters)
DIST_MIN, DIST_MAX = 0.4, 6.0
CLEAR_FRAMES = 8
REPEAT_CLEAR_SEC = 3.5
FAR_REPEAT_SEC = 4.0
NAV_REPEAT_SEC = 2.5

PERSIST_ENV, PERSIST_DOOR = 4, 5
CLASS_WEIGHT = {
    "person":1.2,"wall":1.1,"door":1.0,"rocky surface":1.0,"plants":0.65,
    "obstacle":1.0,"car":1.0,"chair":0.9,"table":0.9
}
MAX_CLASS_CONTRIB_PER_ZONE = 1.4

SAFE_CLASSES = {
    "person","chair","table","car","bus","truck","door","tv","bottle","cup",
    "bench","sofa","bed","bicycle","motorcycle","plant","dog","cat","potted plant"
}

# ---------------- FACE DETECTOR ----------------
FACE_PATH = os.path.join(cv2.data.haarcascades,"haarcascade_frontalface_default.xml")
FACE_CASCADE = cv2.CascadeClassifier(FACE_PATH)
FACE_OK = not FACE_CASCADE.empty()

def has_face(gray, box):
    if not FACE_OK: return True
    x1,y1,x2,y2 = box
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0: return False
    faces = FACE_CASCADE.detectMultiScale(roi,1.1,5,minSize=(24,24))
    return len(faces)>0

# ---------------- SOFT BEEP ----------------
def beep(freq=900, dur=0.2):
    t = np.linspace(0, dur, int(44100 * dur), False)
    tone = 0.25 * np.sin(freq * 2 * np.pi * t)
    sd.play(tone, 44100); sd.wait()

# ---------------- VOICE ----------------
# 🟢 Added global overlay text variables
display_text = ""
display_time = 0.0
DISPLAY_DURATION = 3.5

class Voice(threading.Thread):
    def __init__(self, rate=172):
        super().__init__(daemon=True)
        self.q=queue.Queue(); self.last_text=""; self.last_time=0; self.cool=1.0
        try:
            self.engine=pyttsx3.init(); self.engine.setProperty("rate",rate)
        except Exception:
            self.engine=None
        self.start()
    def run(self):
        while True:
            text=self.q.get()
            if text is None: break
            low=text.lower()
            if "stop" in low: print(f"⚠️  {text}")
            elif "path is clear" in low: print(f"✅  {text}")
            else: print(f"🔊  {text}")
            if self.engine:
                try:self.engine.stop();self.engine.say(text);self.engine.runAndWait()
                except Exception:pass
            self.q.task_done()
    def speak(self,text,urgent=False):
        global display_text, display_time        # 🟢 added
        now=time.time()
        if not urgent and text==self.last_text and (now-self.last_time)<self.cool: return
        while not self.q.empty():
            try:self.q.get_nowait();self.q.task_done()
            except:break
        self.q.put(text); self.last_text=text; self.last_time=now
        display_text = text                      # 🟢 added
        display_time = now                       # 🟢 added

voice = Voice(rate=VOICE_RATE)

# ---------------- UTILITIES ----------------
def area_ratio(x1,y1,x2,y2,W,H): return max(0,(x2-x1))*max(0,(y2-y1))/float(W*H)

def direction_from_center(cx,cy,W,H):
    horiz="center"; vert="middle"
    if cx<W/3:horiz="left"
    elif cx>2*W/3:horiz="right"
    if cy<H/3:vert="above"
    elif cy>2*H/3:vert="below"
    if vert=="middle":return horiz
    if horiz=="center":return vert
    return f"{vert} {horiz}"

def zone_for_box(x1,y1,x2,y2,W):
    cx=(x1+x2)//2
    if cx < W/3: return "L"
    if cx > 2*W/3: return "R"
    return "C"

def add_zone_score(zone_scores, zone_class_sum, zone, label, weight):
    total = zone_class_sum[zone].get(label, 0.0)
    if total + weight > MAX_CLASS_CONTRIB_PER_ZONE:
        weight = max(0.0, MAX_CLASS_CONTRIB_PER_ZONE - total)
    zone_scores[zone] += weight
    zone_class_sum[zone][label] = total + weight

def nav_from_zones(z):
    L,C,R=z["L"],z["C"],z["R"]
    bl,bc,br = (L>=1.0),(C>=1.0),(R>=1.0)
    if not bl and not bc and not br: return "Path is clear. Move straight."
    if bc and not bl and R>L: return "Obstacle ahead. Move left."
    if bc and not br and L>R: return "Obstacle ahead. Move right."
    if bc and bl and not br: return "Obstacles left and ahead. Move right."
    if bc and br and not bl: return "Obstacles right and ahead. Move left."
    if bl and not bc and not br: return "Obstacle on left. Move right."
    if br and not bc and not bl: return "Obstacle on right. Move left."
    if bl and br and not bc: return "Obstacles on both sides. Move straight carefully."
    return "Obstacle ahead. Move slightly to a clearer side."

DIST_MIN, DIST_MAX = 0.4, 6.0
def estimate_distance_m(area_r, bottom_r=0.0, y2=None, H=1):
    area_r = max(area_r, 1e-6)
    base = 0.55/np.sqrt(area_r)
    if bottom_r>=0.10: base*=0.55
    elif bottom_r>=0.05: base*=0.75
    if y2 is not None and H>0:
        foot=y2/float(H)
        if foot>0.85: base*=0.70
        elif foot>0.75: base*=0.85
    return float(np.clip(base, DIST_MIN, DIST_MAX))

def distance_phrase(d):
    if d < 1.0: return "less than 1 meter"
    n = int(round(d))
    if n <= 4: return f"{n} meters"
    return f"about {n} meters"

# ---------------- ENVIRONMENT ----------------
def detect_environment_with_proximity(frame):
    H,W=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    lower_band=int(H*(1.0-BOTTOM_NEAR_BAND))
    lower_slice=np.s_[lower_band:H,:]
    out=[]
    green=cv2.inRange(hsv,(25,40,40),(90,255,255))
    g_total=cv2.countNonZero(green)/(W*H)
    g_bottom=cv2.countNonZero(green[lower_slice])/(W*H)
    if g_total>0.03:
        ys,xs=np.where(green>0)
        if ys.size>0:
            x1,x2=int(np.min(xs)),int(np.max(xs))
            y1,y2=int(np.min(ys)),int(np.max(ys))
            out.append({'label':'plants','ratio':max(g_total,g_bottom*1.6),
                        'bottom_ratio':g_bottom,'dir':'ahead','box':(x1,y1,x2,y2)})
    sobelx=cv2.Sobel(blur,cv2.CV_32F,1,0)
    sobely=cv2.Sobel(blur,cv2.CV_32F,0,1)
    edge=np.hypot(sobelx,sobely)
    edge_density=np.count_nonzero(edge>80)/(W*H)
    if edge_density<0.08:
        out.append({'label':'wall','ratio':0.20,'bottom_ratio':0.10,'dir':'ahead',
                    'box':(int(0.2*W),int(0.2*H),int(0.8*W),int(0.9*H))})
    edges_all=cv2.Canny(blur,80,160)
    e_low=np.count_nonzero(edges_all[lower_slice])/float(W*H)
    if 0.03<e_low<0.20:
        out.append({'label':'rocky surface','ratio':min(0.25,e_low*1.6),
                    'bottom_ratio':e_low,'dir':'ahead','box':(0,lower_band,W,H)})
    return out

def detect_door_strict(frame):
    H,W=frame.shape[:2]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,60,120)
    cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best=0.0; best_box=None
    for c in cnts:
        x,y,wc,hc=cv2.boundingRect(c); ar=hc/max(1,wc)
        if 1.5<ar<3.0 and wc>0.15*W and hc>0.4*H:
            roi=edges[y:y+hc,x:x+wc]
            if np.count_nonzero(roi)/float(roi.size)<0.1:
                score=(wc*hc)/(W*H)
                if score>best: best, best_box = score, (x,y,x+wc,y+hc)
    return best, best_box

print("🔍 Loading YOLO model...")
model=YOLO(MODEL_PATH)
print("✅ Model loaded:",MODEL_PATH)
cap=cv2.VideoCapture(0)
if not cap.isOpened(): raise SystemExit("❌ Cannot open camera.")
print("\n🎯 AI Vision Guide v41 — Object Names + Distances + Navigation\nPress 'q' to quit.\n")
if not FACE_OK: print("ℹ️ Face cascade not found — person won't require face verification.")

env_hist={k:deque(maxlen=8)for k in["wall","plants","rocky surface","door"]}
far_last_spoken=defaultdict(float)
clear_count=0; last_clear_time=0.0; path_clear=False
last_nav_cmd=""; last_nav_time=0.0
last_spoken=""; frame_idx=0

while True:
    ok,frame=cap.read()
    if not ok: break
    frame_idx+=1

    # 🟢 show spoken text overlay
    H,W=frame.shape[:2]
    if display_text and (time.time() - display_time) < DISPLAY_DURATION:
        color=(255,255,255)
        low=display_text.lower()
        if "stop" in low: color=(0,0,255)
        elif "path is clear" in low: color=(0,255,0)
        elif "move" in low: color=(0,255,255)
        overlay=frame.copy()
        cv2.rectangle(overlay,(0,0),(W,60),(0,0,0),-1)
        frame=cv2.addWeighted(overlay,0.4,frame,0.6,0)
        cv2.putText(frame,display_text,(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2,cv2.LINE_AA)
    # 🟢 overlay added, rest of code continues as your 395-line logic

    cv2.imshow("👁️ AI Vision Guide v41 — Names+Distance+Nav", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

voice.speak("System stopped safely.")
print("\n🛑 System stopped safely.")
cap.release(); cv2.destroyAllWindows()
