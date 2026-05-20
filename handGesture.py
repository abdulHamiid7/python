
import sys
import cv2
import pyautogui
import numpy as np
import time

try:
    import mediapipe as mp
  
    _ = mp.solutions.hands
    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    _NEW_API  = False
except AttributeError:
  
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _NEW_API = True

    # ── Download the hand landmarker model if not present
    import urllib.request, os
    _MODEL_PATH = "hand_landmarker.task"
    _MODEL_URL  = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    if not os.path.exists(_MODEL_PATH):
        print(f"[INFO] Downloading MediaPipe hand model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Download complete.")

    _HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),        # thumb
        (0,5),(5,6),(6,7),(7,8),        # index
        (0,9),(9,10),(10,11),(11,12),   # middle
        (0,13),(13,14),(14,15),(15,16), # ring
        (0,17),(17,18),(18,19),(19,20), # pinky
        (5,9),(9,13),(13,17),           # palm
    ]

    def _draw_landmarks_new(frame, landmarks_px):
        """Draw hand skeleton on frame using pixel-space landmark list."""
        for i, (x, y) in enumerate(landmarks_px):
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, landmarks_px[a], landmarks_px[b], (0, 200, 80), 2)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  

SCREEN_W, SCREEN_H = pyautogui.size()

# ── Tunables
SMOOTHING      = 0.18   #
CLICK_COOLDOWN = 0.5    
SCROLL_SPEED   = 30     # 
DEAD_ZONE      = 0.02   # 

# ─────────────────────────────────────────────
# Finger state helpers
# ─────────────────────────────────────────────

def fingers_up(landmarks):
    """
    Returns a list [thumb, index, middle, ring, pinky]
    1 = finger extended, 0 = finger folded.
    """
    tips  = [4, 8, 12, 16, 20]   # landmark indices for fingertips
    bases = [2, 6, 10, 14, 18]   # MCP / pip joints
    state = []

    # Thumb: compare x position (right hand: tip further left = up)
    thumb_tip  = landmarks[tips[0]]
    thumb_ip   = landmarks[tips[0] - 1]
    state.append(1 if thumb_tip.x < thumb_ip.x else 0)

    # Other four fingers: tip y < base y means extended (screen coords, y grows downward)
    for tip, base in zip(tips[1:], bases[1:]):
        state.append(1 if landmarks[tip].y < landmarks[base].y else 0)

    return state   # [thumb, index, middle, ring, pinky]


def landmark_distance(a, b):
    """Euclidean distance between two normalized landmarks."""
    return np.hypot(a.x - b.x, a.y - b.y)


def classify_gesture(fingers, landmarks):
    """
    Returns a gesture name string based on finger states and key distances.
    """
    thumb, index, middle, ring, pinky = fingers

    # Pinch: thumb tip ↔ index tip very close
    pinch_dist = landmark_distance(landmarks[4], landmarks[8])
    if pinch_dist < 0.05:
        return "PINCH"

    # Open hand: all five up
    if all(f == 1 for f in fingers):
        return "OPEN"

    # Fist: none up
    if all(f == 0 for f in fingers):
        return "FIST"

    # Peace / 2-finger click
    if index == 1 and middle == 1 and ring == 0 and pinky == 0:
        return "PEACE"

    # 3 fingers → right click
    if index == 1 and middle == 1 and ring == 1 and pinky == 0:
        return "THREE"

    # Point / move cursor
    if index == 1 and middle == 0 and ring == 0 and pinky == 0:
        return "POINT"

    # Thumb + pinky (shaka) → volume
    if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 1:
        return "SHAKA"

    return "UNKNOWN"


# ─────────────────────────────────────────────
# Main controller class
# ─────────────────────────────────────────────

class GestureController:
    def __init__(self):
        self.prev_x, self.prev_y = SCREEN_W // 2, SCREEN_H // 2
        self.last_click_time = 0
        self.dragging        = False
        self.prev_hand_y     = None   # for scroll / volume reference
        self.gesture_history = []     # short buffer for stability

    # ── Smooth cursor movement
    def move_cursor(self, nx, ny):
        """
        nx, ny are normalized [0,1] coords from the webcam frame.
        Map to screen with exponential smoothing.
        """
        # Flip x (mirror) and map to screen
        target_x = int((1 - nx) * SCREEN_W)
        target_y = int(ny        * SCREEN_H)

        # Exponential moving average
        sx = int(self.prev_x + (target_x - self.prev_x) * SMOOTHING)
        sy = int(self.prev_y + (target_y - self.prev_y) * SMOOTHING)

        # Clamp
        sx = max(0, min(SCREEN_W - 1, sx))
        sy = max(0, min(SCREEN_H - 1, sy))

        pyautogui.moveTo(sx, sy)
        self.prev_x, self.prev_y = sx, sy
        return sx, sy

    def can_click(self):
        return (time.time() - self.last_click_time) > CLICK_COOLDOWN

    def do_click(self, button="left"):
        if self.can_click():
            pyautogui.click(button=button)
            self.last_click_time = time.time()

    # ── Stable gesture over last N frames
    def stable_gesture(self, gesture, n=4):
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > n:
            self.gesture_history.pop(0)
        return self.gesture_history.count(gesture) >= n - 1

    # ── Process one frame
    def handle(self, gesture, hand_center, landmarks):
        cx, cy = hand_center   # normalized

        if gesture == "POINT":
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            self.move_cursor(cx, cy)
            self.prev_hand_y = cy

        elif gesture == "PINCH":
            sx, sy = self.move_cursor(cx, cy)
            if not self.dragging and self.stable_gesture("PINCH"):
                pyautogui.mouseDown()
                self.dragging = True

        elif gesture == "PEACE":
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            if self.stable_gesture("PEACE"):
                self.do_click("left")

        elif gesture == "THREE":
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            if self.stable_gesture("THREE"):
                self.do_click("right")

        elif gesture == "OPEN":
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            # Scroll: compare current y to previous
            if self.prev_hand_y is not None:
                delta = cy - self.prev_hand_y
                if abs(delta) > DEAD_ZONE:
                    scroll_amt = int(-delta * SCROLL_SPEED * 20)
                    pyautogui.scroll(scroll_amt)
            self.prev_hand_y = cy

        elif gesture == "SHAKA":
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            if self.prev_hand_y is not None:
                delta = self.prev_hand_y - cy  # up = positive = volume up
                if abs(delta) > DEAD_ZONE:
                    if delta > 0:
                        pyautogui.press("volumeup")
                    else:
                        pyautogui.press("volumedown")
            self.prev_hand_y = cy

        elif gesture == "FIST":
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            self.prev_hand_y = cy

        else:
            self.prev_hand_y = cy


# ─────────────────────────────────────────────
# Draw HUD overlay
# ─────────────────────────────────────────────

GESTURE_COLORS = {
    "POINT":   (0, 255, 100),
    "PEACE":   (0, 200, 255),
    "THREE":   (255, 150, 0),
    "OPEN":    (255, 255, 0),
    "PINCH":   (255, 80, 200),
    "SHAKA":   (80, 160, 255),
    "FIST":    (100, 100, 100),
    "UNKNOWN": (60, 60, 60),
}

GESTURE_ICONS = {
    "POINT":   "☝  Move Cursor",
    "PEACE":   "✌  Left Click",
    "THREE":   "🤟 Right Click",
    "OPEN":    "🖐  Scroll",
    "PINCH":   "👌 Drag",
    "SHAKA":   "🤙 Volume",
    "FIST":    "✊ Neutral",
    "UNKNOWN": "?  Unknown",
}

def draw_hud(frame, gesture, fps, cursor_pos, fingers):
    h, w = frame.shape[:2]
    color = GESTURE_COLORS.get(gesture, (80, 80, 80))

    # Semi-transparent sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Title
    cv2.putText(frame, "GESTURE CONTROL AI", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:>5.1f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # Cursor
    cx, cy = cursor_pos
    cv2.putText(frame, f"Cursor: ({cx}, {cy})", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Finger indicators
    labels = ["T", "I", "M", "R", "P"]
    for i, (lbl, val) in enumerate(zip(labels, fingers)):
        fc = (0, 255, 120) if val else (60, 60, 80)
        cv2.circle(frame, (20 + i * 42, 108), 14, fc, -1)
        cv2.putText(frame, lbl, (12 + i * 42, 113),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    # Active gesture box
    cv2.rectangle(frame, (8, 128), (272, 175), color, -1)
    label = GESTURE_ICONS.get(gesture, gesture)
    cv2.putText(frame, label, (12, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

    # Legend
    y0 = 195
    cv2.putText(frame, "GESTURES", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
    for i, (g, icon) in enumerate(GESTURE_ICONS.items()):
        yy = y0 + 20 + i * 20
        dot_color = GESTURE_COLORS[g]
        cv2.circle(frame, (16, yy - 4), 5, dot_color, -1)
        cv2.putText(frame, icon, (26, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200) if g == gesture else (120, 120, 120), 1)

    # Quit hint
    cv2.putText(frame, "Press Q to quit", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    return frame


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

class _NormLandmark:
    """Thin wrapper so new-API landmarks look like legacy NormalizedLandmark."""
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    controller = GestureController()

    # ── Build the detector (API-agnostic)
    if not _NEW_API:
        # Legacy mediapipe < 0.10
        detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.65,
        )
    else:
        # New mediapipe >= 0.10 Tasks API
        _base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        _opts = mp_vision.HandLandmarkerOptions(
            base_options=_base_opts,
            num_hands=1,
            min_hand_detection_confidence=0.75,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.65,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        detector = mp_vision.HandLandmarker.create_from_options(_opts)

    prev_time  = time.time()
    cursor_pos = (SCREEN_W // 2, SCREEN_H // 2)
    gesture    = "FIST"
    fingers    = [0, 0, 0, 0, 0]
    frame_idx  = 0

    print("=" * 50)
    print("  Hand Gesture Control AI — running!")
    print(f"  MediaPipe API: {'NEW (0.10+)' if _NEW_API else 'LEGACY (0.9.x)'}")
    print("  Move mouse to top-left corner to quit (failsafe)")
    print("  Or press Q in the camera window")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame.")
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lm_list     = None
        lm_list_px  = None  # pixel coords for drawing (new API only)

        if not _NEW_API:
            # ── Legacy path ──────────────────────────────────
            results = detector.process(rgb)
            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                lm_list = hand_lm.landmark
        else:
            # ── New Tasks API path ───────────────────────────
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int(time.time() * 1000)
            result   = detector.detect_for_video(mp_image, ts_ms)
            if result.hand_landmarks:
                raw = result.hand_landmarks[0]   # list of NormalizedLandmark
                lm_list = [_NormLandmark(lm.x, lm.y, lm.z) for lm in raw]
                lm_list_px = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
                _draw_landmarks_new(frame, lm_list_px)

        if lm_list is not None:
            fingers     = fingers_up(lm_list)
            gesture     = classify_gesture(fingers, lm_list)
            wrist       = lm_list[0]
            hand_center = (wrist.x, wrist.y)
            cursor_pos  = controller.move_cursor(*hand_center) if gesture != "FIST" else cursor_pos
            controller.handle(gesture, hand_center, lm_list)
        else:
            gesture = "FIST"
            fingers = [0, 0, 0, 0, 0]
            controller.gesture_history.clear()

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now
        frame_idx += 1

        frame = draw_hud(frame, gesture, fps, cursor_pos, fingers)
        cv2.imshow("Hand Gesture Control AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if not _NEW_API:
        detector.close()
    else:
        detector.close()
    cap.release()
    cv2.destroyAllWindows()

    if controller.dragging:
        pyautogui.mouseUp()

    print("Gesture controller stopped.")


if __name__ == "__main__":
    main()
