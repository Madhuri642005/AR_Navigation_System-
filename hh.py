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
        now=time.time()
        if not urgent and text==self.last_text and (now-self.last_time)<self.cool: return
        while not self.q.empty():
            try:self.q.get_nowait();self.q.task_done()
            except:break
        self.q.put(text); self.last_text=text; self.last_time=now

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

# Distance estimate (meters), then phrase it for speech
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
    # >4 m — say "about N meters"
    return f"about {n} meters"

# ---------------- ENVIRONMENT ----------------
def detect_environment_with_proximity(frame):
    """
    Returns list of dicts:
      {'label','ratio','bottom_ratio','dir','box'}
    """
    H,W=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    lower_band=int(H*(1.0-BOTTOM_NEAR_BAND))
    lower_slice=np.s_[lower_band:H,:]
    out=[]

    # Plants
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

    # Wall (flatness proxy)
    sobelx=cv2.Sobel(blur,cv2.CV_32F,1,0)
    sobely=cv2.Sobel(blur,cv2.CV_32F,0,1)
    edge=np.hypot(sobelx,sobely)
    edge_density=np.count_nonzero(edge>80)/(W*H)
    if edge_density<0.08:
        out.append({'label':'wall','ratio':0.20,'bottom_ratio':0.10,'dir':'ahead',
                    'box':(int(0.2*W),int(0.2*H),int(0.8*W),int(0.9*H))})

    # Rocky (roughness bottom)
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

# ---------------- LOAD YOLO + CAMERA ----------------
print("🔍 Loading YOLO model...")
model=YOLO(MODEL_PATH)
print("✅ Model loaded:",MODEL_PATH)
cap=cv2.VideoCapture(0)
if not cap.isOpened(): raise SystemExit("❌ Cannot open camera.")
print("\n🎯 AI Vision Guide v41 — Object Names + Distances + Navigation\nPress 'q' to quit.\n")
if not FACE_OK: print("ℹ️ Face cascade not found — person won't require face verification.")

# ---------------- STATE ----------------
env_hist={k:deque(maxlen=8)for k in["wall","plants","rocky surface","door"]}
far_last_spoken=defaultdict(float)
clear_count=0; last_clear_time=0.0; path_clear=False
last_nav_cmd=""; last_nav_time=0.0
last_spoken=""; frame_idx=0

# ---------------- MAIN LOOP ----------------
while True:
    ok,frame=cap.read()
    if not ok: break
    frame_idx+=1
    if frame_idx%FRAME_SKIP!=0:
        cv2.imshow("AI Vision Guide v41",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):break
        continue

    H,W=frame.shape[:2]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    now=time.time()

    msgs=[]; stop=False; stop_msg=""
    zone_scores={"L":0.0,"C":0.0,"R":0.0}
    zone_class_sum={"L":defaultdict(float),"C":defaultdict(float),"R":defaultdict(float)}
    blockers={"L":[], "C":[], "R":[]}

    # ---------- YOLO OBJECTS ----------
    results=model(frame,conf=CONF_OBJ,verbose=False)
    for r in results:
        if getattr(r,"boxes",None) is None: continue
        for b in r.boxes:
            cls=int(b.cls[0]); label=model.names[cls]
            conf=float(b.conf[0]); x1,y1,x2,y2=map(int,b.xyxy[0].tolist())
            bw,bh=x2-x1,y2-y1
            if bw<40 or bh<40: continue
            if label=="person" and conf<CONF_PERSON: continue
            if label!="person" and conf<CONF_OBJ: continue
            if label=="person" and FACE_OK and not has_face(gray,(x1,y1,x2,y2)): continue

            ar=area_ratio(x1,y1,x2,y2,W,H)
            cx,cy=(x1+x2)//2,(y1+y2)//2
            dirn=direction_from_center(cx,cy,W,H)
            z=zone_for_box(x1,y1,x2,y2,W)
            lbl=label if label in SAFE_CLASSES else "obstacle"

            d = estimate_distance_m(ar, 0.0, y2, H)
            d_phrase = distance_phrase(d)

            # STOP
            if ar >= STOP_RATIO:
                stop=True; stop_msg=f"Stop please! {lbl} very close {dirn}, {d_phrase}."
            # zone risk
            base_w = min(1.0, ar*5.0)*CLASS_WEIGHT.get(lbl,1.0)
            add_zone_score(zone_scores, zone_class_sum, z, lbl, base_w)
            blockers[z].append((lbl, d_phrase))

            # info message (with distance)
            if ar >= 0.03:  # speak for sensible sizes
                if ar < STOP_RATIO:
                    # far/near phrasing included in d_phrase
                    # Reassure if far/very far (d >= ~3 m -> phrase gives "about N m")
                    if now - far_last_spoken[f"obj:{lbl}"] >= FAR_REPEAT_SEC:
                        msgs.append((ar, f"{lbl} {d_phrase} {dirn}. Path is clear up to that distance."))
                        far_last_spoken[f"obj:{lbl}"] = now

    # ---------- ENVIRONMENT ----------
    env_list = detect_environment_with_proximity(frame)
    door_score, door_box = detect_door_strict(frame)
    if door_score>0 and door_box is not None:
        x1,y1,x2,y2 = door_box
        env_list.append({'label':'door','ratio':min(0.25,door_score),'bottom_ratio':0.0,
                         'dir':'ahead','box':door_box})

    # persistence
    present={e['label'] for e in env_list}
    env_hist.setdefault("door", deque(maxlen=8))
    for k in env_hist: env_hist[k].append(1 if k in present else 0)
    def persistent(lbl): 
        need = PERSIST_DOOR if lbl=="door" else PERSIST_ENV
        return sum(env_hist.get(lbl, deque())) >= need

    for e in env_list:
        lbl, ratio, br, dirn = e['label'], e['ratio'], e['bottom_ratio'], e['dir']
        if not persistent(lbl): continue
        x1,y1,x2,y2 = e['box']
        z=zone_for_box(x1,y1,x2,y2,W)
        d = estimate_distance_m(ratio, br, y2, H)
        d_phrase = distance_phrase(d)

        # STOP by bottom proximity or closeness
        if br >= BOTTOM_FORCE_STOP_RATIO or ratio >= STOP_RATIO:
            stop=True; stop_msg=f"Stop please! {lbl} very close {dirn}, {d_phrase}."
        # zone risk
        base_w = min(1.0, ratio*5.0)*CLASS_WEIGHT.get(lbl,1.0)
        add_zone_score(zone_scores, zone_class_sum, z, lbl, base_w)
        blockers[z].append((lbl, d_phrase))

        # info line with distance
        if now - far_last_spoken[f"env:{lbl}"] >= FAR_REPEAT_SEC:
            msgs.append((ratio, f"{lbl} {d_phrase} {dirn}. Path is clear up to that distance."))
            far_last_spoken[f"env:{lbl}"] = now

    # ---------- DECISION & SPEECH ----------
    if stop:
        beep()
        voice.speak(stop_msg, urgent=True)
        last_spoken = stop_msg
        path_clear=False; clear_count=0
        last_nav_cmd=""; last_nav_time=now
    else:
        # Build navigation sentence with reasons + distances
        nav_cmd = nav_from_zones(zone_scores)

        def reason_phrase(side):
            if not blockers[side]: return ""
            # pick up to 2 distinct classes with distances
            seen=set(); parts=[]
            for lbl, dph in blockers[side]:
                if lbl in seen: continue
                parts.append(f"{lbl} {dph}")
                seen.add(lbl)
                if len(parts)==2: break
            return ", ".join(parts)

        if "Move left" in nav_cmd and blockers["C"]:
            cause = reason_phrase("C")
            if cause: nav_cmd = f"Obstacle ahead due to {cause}. Move left."
        elif "Move right" in nav_cmd and blockers["C"]:
            cause = reason_phrase("C")
            if cause: nav_cmd = f"Obstacle ahead due to {cause}. Move right."
        elif "Obstacle on left" in nav_cmd and blockers["L"]:
            cause = reason_phrase("L")
            nav_cmd = f"Obstacle on left due to {cause}. Move right."
        elif "Obstacle on right" in nav_cmd and blockers["R"]:
            cause = reason_phrase("R")
            nav_cmd = f"Obstacle on right due to {cause}. Move left."
        elif "Move straight carefully" in nav_cmd:
            causeL = reason_phrase("L"); causeR = reason_phrase("R")
            cause = ", ".join([c for c in [causeL,causeR] if c])
            if cause: nav_cmd = f"Obstacles on sides ({cause}). Move straight carefully."

        # Speak navigation with pacing
        if (nav_cmd != last_nav_cmd) or (now - last_nav_time >= NAV_REPEAT_SEC):
            voice.speak(nav_cmd)
            last_spoken = nav_cmd
            last_nav_cmd = nav_cmd; last_nav_time = now

        # Path-clear reassurance (only when truly clear)
        if nav_cmd.startswith("Path is clear"):
            clear_count += 1
            if clear_count >= CLEAR_FRAMES:
                if (not path_clear) or (now - last_clear_time >= REPEAT_CLEAR_SEC):
                    voice.speak("Path is clear. Move straight.")
                    last_spoken = "Path is clear. Move straight."
                    last_clear_time = now; path_clear = True
        else:
            path_clear=False; clear_count=0

        # Also speak one top info message (far/near with distance)
        if msgs:
            msgs.sort(key=lambda x:-x[0])
            info_msg = msgs[0][1]
            if info_msg != last_spoken and (now - voice.last_time) > 0.6:
                voice.speak(info_msg)
                last_spoken = info_msg

    # ---------- DISPLAY ----------
    cv2.imshow("👁️ AI Vision Guide v41 — Names+Distance+Nav", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

voice.speak("System stopped safely.")
print("\n🛑 System stopped safely.")
cap.release(); cv2.destroyAllWindows()                  