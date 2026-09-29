from ultralytics import YOLO
import cv2, numpy as np, pyttsx3, sounddevice as sd, time, threading

class VisionGuide:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)
        self.stop_ratio = 0.25
        self.last_msg = ""
        self.last_time = 0
        self.tik_running = False

    def tik_tik(self):
        """Continuous tik-tik when too near"""
        self.tik_running = True
        while self.tik_running:
            t = np.linspace(0, 0.08, int(44100 * 0.08), False)
            tone = 0.4 * np.sin(2 * np.pi * 900 * t)
            sd.play(tone, 44100)
            sd.wait()
            time.sleep(0.15)

    def stop_tik(self):
        self.tik_running = False

    def speak(self, msg):
        now = time.time()
        if msg == self.last_msg and (now - self.last_time) < 1.2:
            return
        self.engine.say(msg)
        self.engine.runAndWait()
        self.last_msg = msg
        self.last_time = now

    def detect(self, frame):
        H, W = frame.shape[:2]
        results = self.model(frame, conf=0.6, verbose=False)
        stop = False
        msg = "Path is clear. Move straight."

        for r in results:
            if getattr(r, "boxes", None) is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1) / (W * H)
                distance = round(1 / np.sqrt(max(area, 1e-6)), 1)
                color = (0, 255, 0)

                if area >= self.stop_ratio:
                    stop = True
                    color = (0, 0, 255)
                    msg = f"Stop! {label} is very close, less than 1 meter."
                    if not self.tik_running:
                        threading.Thread(target=self.tik_tik).start()
                else:
                    msg = f"{label} about {distance} meters ahead."
                    self.stop_tik()

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {distance}m", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.speak(msg)
        return frame, stop, msg
