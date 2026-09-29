import os
import time
import math
from collections import defaultdict
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from ultralytics import YOLO
import openrouteservice

# ======================================================
# GPS Helper (same small JS approach)
# ======================================================
def geolocation():
    components.html(
        """
        <script>
        function sendLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    function(position) {
                        const coords = {latitude: position.coords.latitude, longitude: position.coords.longitude};
                        window.parent.postMessage({ type: "streamlit:setComponentValue", key: "geo", value: coords }, "*");
                    },
                    function(error) { console.log("Location error:", error); }
                );
            }
        }
        sendLocation();
        </script>
        """,
        height=0,
    )
    return st.session_state.get("geo", None)

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(page_title="AI Vision Guide v47.2 (Windows)", layout="wide")
st.title("👁️ AI Vision Guide v47.2 — Global Navigation + Object Detection (Windows)")

with st.sidebar:
    st.header("⚙️ Settings")
    ors_key = st.text_input("🌍 OpenRouteService API Key", value=os.getenv("ORS_API_KEY", ""))
    model_choice = st.selectbox("YOLO Model", ["yolov8m.pt", "yolov8s.pt", "yolov8n.pt"], index=1)
    conf_obj = st.slider("Object confidence", 0.1, 0.9, 0.55, 0.01)
    conf_person = st.slider("Person min confidence", 0.1, 0.95, 0.80, 0.01)
    iou_nms = st.slider("IOU NMS", 0.1, 0.9, 0.45, 0.01)
    frame_skip = st.slider("Process every Nth frame", 1, 5, 2, 1)
    st.markdown("---")
    st.caption("✅ Works globally. You can use manual or GPS source.")

# ======================================================
# JS SPEECH
# ======================================================
speech_js = """
<script>
window.voiceQueue = window.voiceQueue || [];
window.sayNow = (txt) => {
  const u = new SpeechSynthesisUtterance(txt);
  u.rate = 1.03; u.pitch = 1.0; u.lang = "en-US";
  speechSynthesis.speak(u);
};
const speakLoop = () => {
  if (window.voiceQueue.length > 0 && !speechSynthesis.speaking) {
    const t = window.voiceQueue.shift();
    window.sayNow(t);
  }
  requestAnimationFrame(speakLoop);
};
speakLoop();
window.addEventListener("message", (ev) => {
  if (ev.data && ev.data.type === "speak") window.voiceQueue.push(ev.data.text);
});
</script>
"""
st.components.v1.html(speech_js, height=0)

def speak(text):
    st.components.v1.html(f"""
    <script>
      window.postMessage({{"type":"speak", "text": {repr(text)}}}, "*");
    </script>
    """, height=0)

# ======================================================
# NAVIGATION SECTION (unchanged logic)
# ======================================================
st.subheader("🧭 Navigation System (Global Routing)")

col1, col2 = st.columns(2)
with col1:
    st.write("**Your Location (Auto GPS or Manual)**")
    loc = geolocation()
    src_coords = None
    gps_ok = False
    if loc and loc.get("latitude") and loc.get("longitude"):
        src_coords = (loc["longitude"], loc["latitude"])
        gps_ok = True
        st.success(f"📍 GPS: lat {loc['latitude']:.6f}, lon {loc['longitude']:.6f}")
    else:
        st.warning("⚠️ GPS not available. You can use manual source below.")

    use_manual_src = st.checkbox("Enter source manually (if GPS not working)")
    manual_src_text = ""
    if use_manual_src:
        manual_src_text = st.text_input("Manual source (e.g., 'Shivaji Circle Nipani' or 'Majestic Bangalore')", "")

with col2:
    dest_input = st.text_input("Destination (e.g., 'Bus Stand Nipani' or 'MG Road Bangalore')", "")
    start_nav = st.button("Start Navigation 🚀")

route_status = st.empty()
route_steps = st.empty()


def ors_geocode(place, key):
    client = openrouteservice.Client(key=key)
    res = client.pelias_search(text=place)
    if "features" in res and len(res["features"]) > 0:
        coords = res["features"][0]["geometry"]["coordinates"]
        return tuple(coords)
    return None


def ors_route(src, dest, key):
    client = openrouteservice.Client(key=key)
    route = client.directions(
        coordinates=[src, dest],
        profile="foot-walking",
        format="geojson"
    )
    steps = []
    for f in route["features"]:
        for seg in f["properties"]["segments"]:
            for s in seg["steps"]:
                steps.append(f"{s['instruction']} — {int(s['distance'])} meters")
    return steps

if start_nav:
    if not ors_key:
        route_status.error("❌ Please add your OpenRouteService API key in the sidebar.")
    elif not dest_input.strip():
        route_status.warning("Enter a destination name.")
    else:
        # Resolve source
        if not gps_ok:
            if use_manual_src and manual_src_text.strip():
                src_coords = ors_geocode(manual_src_text.strip(), ors_key)
                if not src_coords:
                    route_status.error("❌ Could not find the manual source. Try a nearby landmark.")
            else:
                route_status.warning("⚠️ GPS not ready. Either allow location or type your source manually.")
        # Resolve destination
        dest_coords = None
        if src_coords:
            dest_coords = ors_geocode(dest_input.strip(), ors_key)
            if not dest_coords:
                route_status.error("❌ Could not find destination. Check spelling or try a nearby landmark.")

        # Fetch route
        if src_coords and dest_coords:
            try:
                steps = ors_route(src_coords, dest_coords, ors_key)
                if steps:
                    route_status.success(f"🗺️ Route ready to {dest_input}")
                    with route_steps.container():
                        st.markdown("### 🚶 Voice Route Directions")
                        for s in steps:
                            st.write("• " + s)
                    speak(f"Starting navigation to {dest_input}. Follow the spoken directions.")
                    for s in steps[:5]:
                        speak(s)
                else:
                    route_status.warning("⚠️ No walking route found.")
            except Exception as e:
                route_status.error(f"Routing error: {e}")

st.markdown("---")

# ======================================================
# OBJECT DETECTION SECTION (OpenCV-based for Windows)
# ======================================================
st.subheader("📷 Object Detection + Voice Feedback (OpenCV)")

STOP_RATIO = 0.27
DIST_MIN, DIST_MAX = 0.4, 6.0
SAFE_CLASSES = {"person","chair","table","car","bus","truck","door","tv","bottle","cup","bench","sofa","bed","bicycle","motorcycle","plant","dog","cat","potted plant"}


def distance_phrase(d):
    if d < 1.0: return "less than 1 meter"
    n = int(round(d))
    if n <= 4: return f"{n} meters"
    return f"about {n} meters"


def estimate_distance(area_ratio):
    area_ratio = max(1e-6, area_ratio)
    base = 0.55 / math.sqrt(area_ratio)
    return float(np.clip(base, DIST_MIN, DIST_MAX))


def direction_from_center(cx, cy, W, H):
    horiz = "center"
    if cx < W/3: horiz = "left"
    elif cx > 2*W/3: horiz = "right"
    if cy > 2*H/3:
        return "below" if horiz == "center" else f"below {horiz}"
    return horiz


def area_ratio_xyxy(x1,y1,x2,y2,W,H):
    return max(0,(x2-x1))*max(0,(y2-y1))/float(max(1,W*H))

# Load YOLO model once
_model = YOLO(model_choice)
st.success(f"✅ YOLO model loaded: {model_choice}")

# UI controls for camera and detection
col_cam, col_ctrl = st.columns([3,1])
with col_ctrl:
    start_cam = st.checkbox("Start Camera")
    use_dshow = st.checkbox("Use DirectShow backend (Windows) (recommended)", value=True)
    show_fps = st.checkbox("Show FPS", value=True)

frame_window = st.image([])
log_box = st.empty()

# state for speaking cooldowns
last_speak_time = defaultdict(float)

# camera capture
cap = None

try:
    if start_cam:
        backend = cv2.CAP_DSHOW if use_dshow else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            st.error("Could not open camera. Make sure no other app is using it and you have allowed permissions.")
            start_cam = False
        else:
            st.info("Camera started. Processing frames...")

    prev = 0
    frame_idx = 0
    fps = 0.0

    while start_cam and cap and cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            log_box.warning("Failed to read frame from camera")
            break

        H, W = frame.shape[:2]
        frame_idx += 1

        # optionally skip frames
        if frame_idx % frame_skip == 0:
            # run YOLO
            results = _model.predict(frame, conf=conf_obj, iou=iou_nms, verbose=False)
            if len(results) > 0:
                res = results[0]
                if hasattr(res, 'boxes') and res.boxes is not None:
                    for b in res.boxes:
                        cls = int(b.cls.item())
                        label = _model.names[cls]
                        conf = float(b.conf.item())
                        if label == "person" and conf < conf_person:
                            continue

                        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                        ar = area_ratio_xyxy(x1,y1,x2,y2,W,H)
                        if ar < 1e-5:
                            continue
                        d = estimate_distance(ar)
                        dirn = direction_from_center((x1+x2)//2,(y1+y2)//2,W,H)
                        lbl = label if label in SAFE_CLASSES else "obstacle"

                        if ar >= STOP_RATIO:
                            speak(f"Stop! {lbl} very close {dirn}, {distance_phrase(d)}.")
                        elif ar >= 0.02:
                            now = time.time()
                            key = f"{lbl}:{dirn}"
                            if now - last_speak_time[key] > 3.5:
                                speak(f"{lbl} {distance_phrase(d)} {dirn}.")
                                last_speak_time[key] = now

                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                        cv2.putText(frame, f"{lbl} {distance_phrase(d)}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # show FPS
        t1 = time.time()
        dt = t1 - prev if prev else 0.0
        prev = t1
        fps = 1.0/dt if dt > 0 else fps
        if show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # convert BGR->RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

        # small sleep to yield to Streamlit and avoid 100% CPU
        time.sleep(0.02)

        # update start_cam in case user toggled off in the UI
        start_cam = st.session_state.get('Start Camera', start_cam)

finally:
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    st.info("Camera stopped (if it was running).")
