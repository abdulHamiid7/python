"""
AI Virtual Mouse — Python + MediaPipe Tasks API (v0.10.14+)
============================================================
Works with mediapipe 0.10.14 and above (including 0.10.35).
Uses the new HandLandmarker Tasks API — no mp.solutions needed.

GESTURES:
  • Index finger up              → Move cursor
  • Index + Middle up, pinch     → Left click
  • Thumb + Index pinch          → Right click
  • Fist (all fingers closed)    → Drag
  • Index + Middle + Ring up     → Scroll (move hand up/down)

INSTALL (run once):
  pip install opencv-python mediapipe pyautogui numpy

RUN:
  python virtual_mouse.py
  Press  Q  to quit.
  Move mouse to top-left corner = emergency stop (PyAutoGUI failsafe).
"""

import cv2
import pyautogui
import numpy as np
import time
import urllib.request
import os

# ── MediaPipe Tasks API (works on 0.10.14 → 0.10.35+) ───────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX     = 0      # Change to 1/2 if wrong camera opens
FRAME_REDUCTION  = 100    # Active-zone border (pixels)
SMOOTHING        = 5      # Rolling average window (1–10)
CLICK_THRESHOLD  = 40     # Pixel distance to fire a click
SCROLL_SPEED     = 15     # Scroll sensitivity
MODEL_PATH       = "hand_landmarker.task"   # Auto-downloaded if missing
# ──────────────────────────────────────────────────────────────────────────────

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# ── Landmark indices ──────────────────────────────────────────────────────────
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading hand landmarker model → '{MODEL_PATH}' …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Download complete.")

def fingers_up(landmarks, handedness_label):
    """Return [thumb, index, middle, ring, pinky] booleans."""
    fingers = []
    # Thumb — horizontal (frame is mirrored so Left/Right are swapped)
    if handedness_label == "Left":
        fingers.append(landmarks[4].x < landmarks[3].x)
    else:
        fingers.append(landmarks[4].x > landmarks[3].x)
    # Other four fingers — tip.y < pip.y means extended
    for tip, pip in zip(TIP_IDS[1:], PIP_IDS[1:]):
        fingers.append(landmarks[tip].y < landmarks[pip].y)
    return fingers

def px_dist(landmarks, a, b, w, h):
    ax, ay = landmarks[a].x * w, landmarks[a].y * h
    bx, by = landmarks[b].x * w, landmarks[b].y * h
    return np.hypot(ax - bx, ay - by)

def draw_hand(frame, landmarks, w, h):
    """Draw skeleton without mp.solutions.drawing_utils."""
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), cv2.FILLED)
    for tip in TIP_IDS:
        cv2.circle(frame, pts[tip], 7, (0, 100, 255), cv2.FILLED)

# ── Smoother ──────────────────────────────────────────────────────────────────
class Smoother:
    def __init__(self, n):
        self.n = n; self.xs = []; self.ys = []

    def smooth(self, x, y):
        self.xs.append(x); self.ys.append(y)
        if len(self.xs) > self.n: self.xs.pop(0); self.ys.pop(0)
        return int(np.mean(self.xs)), int(np.mean(self.ys))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_model()

    base_opts  = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    hand_opts  = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. "
              "Try changing CAMERA_INDEX at the top of the script.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    screen_w, screen_h = pyautogui.size()
    smoother        = Smoother(SMOOTHING)
    prev_click_time = 0
    dragging        = False
    scroll_ref_y    = None
    frame_ts_ms     = 0

    print("[INFO] Virtual Mouse running — press Q in the camera window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_ts_ms += 33          # must be monotonically increasing
        result = landmarker.detect_for_video(mp_image, frame_ts_ms)

        # Active-zone rectangle
        cv2.rectangle(frame,
                      (FRAME_REDUCTION, FRAME_REDUCTION),
                      (w - FRAME_REDUCTION, h - FRAME_REDUCTION),
                      (0, 200, 100), 2)

        gesture_label = "No hand"

        if result.hand_landmarks:
            lm         = result.hand_landmarks[0]
            handedness = result.handedness[0][0].display_name   # "Left"/"Right"

            draw_hand(frame, lm, w, h)
            fingers = fingers_up(lm, handedness)

            # Map index fingertip to screen
            ix = int(np.interp(lm[8].x * w,
                               [FRAME_REDUCTION, w - FRAME_REDUCTION],
                               [0, screen_w]))
            iy = int(np.interp(lm[8].y * h,
                               [FRAME_REDUCTION, h - FRAME_REDUCTION],
                               [0, screen_h]))
            sx, sy = smoother.smooth(ix, iy)
            tip_px = (int(lm[8].x * w), int(lm[8].y * h))
            cv2.circle(frame, tip_px, 10, (255, 0, 255), cv2.FILLED)

            # ── SCROLL ─────────────────────────────────────────────────────
            if fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
                gesture_label = "Scroll"
                if dragging: pyautogui.mouseUp(); dragging = False
                cur_y = int(lm[8].y * h)
                if scroll_ref_y is None: scroll_ref_y = cur_y
                delta = scroll_ref_y - cur_y
                if abs(delta) > 5:
                    pyautogui.scroll(int(delta / h * SCROLL_SPEED * 10))
                    scroll_ref_y = cur_y

            # ── DRAG (fist) ────────────────────────────────────────────────
            elif not any(fingers[1:]):
                gesture_label = "Drag"
                scroll_ref_y  = None
                if not dragging: pyautogui.mouseDown(); dragging = True
                pyautogui.moveTo(sx, sy)

            # ── RIGHT CLICK (thumb + index pinch) ─────────────────────────
            elif fingers[0] and fingers[1] and not fingers[2]:
                gesture_label = "Right Click"
                scroll_ref_y  = None
                d  = px_dist(lm, 4, 8, w, h)
                p4 = (int(lm[4].x * w), int(lm[4].y * h))
                cv2.line(frame, p4, tip_px, (0, 255, 0), 3)
                if d < CLICK_THRESHOLD:
                    now = time.time()
                    if now - prev_click_time > 0.4:
                        if dragging: pyautogui.mouseUp(); dragging = False
                        pyautogui.rightClick(sx, sy)
                        prev_click_time = now
                        cv2.circle(frame, tip_px, 15, (0, 0, 255), cv2.FILLED)
                else:
                    pyautogui.moveTo(sx, sy)

            # ── LEFT CLICK (index + middle pinch) ─────────────────────────
            elif fingers[1] and fingers[2]:
                gesture_label = "Click / Move"
                scroll_ref_y  = None
                if dragging: pyautogui.mouseUp(); dragging = False
                pyautogui.moveTo(sx, sy)
                d   = px_dist(lm, 8, 12, w, h)
                p12 = (int(lm[12].x * w), int(lm[12].y * h))
                cv2.line(frame, tip_px, p12, (0, 255, 255), 3)
                if d < CLICK_THRESHOLD:
                    now = time.time()
                    if now - prev_click_time > 0.4:
                        pyautogui.click()
                        prev_click_time = now
                        cv2.circle(frame, tip_px, 15, (0, 255, 0), cv2.FILLED)

            # ── MOVE (index only) ──────────────────────────────────────────
            elif fingers[1]:
                gesture_label = "Move"
                scroll_ref_y  = None
                if dragging: pyautogui.mouseUp(); dragging = False
                pyautogui.moveTo(sx, sy)

            else:
                gesture_label = "Idle"
                scroll_ref_y  = None

        else:
            if dragging: pyautogui.mouseUp(); dragging = False
            scroll_ref_y = None

        # HUD overlay
        cv2.putText(frame, f"Gesture: {gesture_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
        cv2.putText(frame, "Q = quit  |  top-left corner = emergency stop",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("AI Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if dragging:
        pyautogui.mouseUp()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Virtual Mouse stopped.")

if __name__ == "__main__":
    main()