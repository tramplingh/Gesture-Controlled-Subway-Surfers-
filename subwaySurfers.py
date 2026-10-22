"""
=============================================================================
  SUBWAY SURFERS — DYNAMIC GESTURE CONTROLLER
  Controls : Lean Left / Right  →  arrow keys left / right
             Jump (stand tall)  →  arrow key up
             Duck (squat)       →  arrow key down

  Stack    : OpenCV  +  MediaPipe Tasks API  +  PyAutoGUI
=============================================================================
"""

import os
import sys
import time
import urllib.request

import cv2
import pyautogui

import mediapipe as mp
from mediapipe.tasks                    import python   as mp_python
from mediapipe.tasks.python             import vision   as mp_vision
from mediapipe.tasks.python.vision      import PoseLandmarker, PoseLandmarkerOptions

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 
# ═════════════════════════════════════════════════════════════════════════════

LANDMARK_CHOICE     = "nose"   # "shoulders" | "nose"
THRESHOLD_X         = 60            # pixels to move left/right to trigger
THRESHOLD_Y         = 60            # pixels to move up/down to trigger
COOLDOWN_SECONDS    = 0.30           # min seconds between keypresses
CAMERA_INDEX        = 0             # 0 = default webcam    

# Visual colours  (BGR)
BOX_COLOR_IDLE  = (0, 220, 0)       # green  — ready to fire
BOX_COLOR_COOL  = (0, 100, 255)     # orange — in cooldown
DOT_COLOR       = (255,  60, 60)    # blue   — tracking point dot
DOT_RADIUS      = 13
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# Model — downloaded automatically on first run
MODEL_FILENAME = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# PyAutoGUI settings
pyautogui.FAILSAFE = True   
pyautogui.PAUSE    = 0.0    


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL BOOTSTRAP
# ═════════════════════════════════════════════════════════════════════════════

def ensure_model() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)

    if not os.path.exists(model_path):
        print(f"[SETUP] Model file not found. Downloading (~6 MB)…")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("[SETUP] Download complete.\n")
        except Exception as exc:
            print(f"\n[ERROR] Could not download model: {exc}")
            sys.exit(1)

    return model_path


# ═════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def get_tracking_point(landmarks: list, frame_w: int, frame_h: int):
    def is_visible(lm) -> bool:
        v = getattr(lm, "visibility", None)
        return (v is None) or (v >= 0.5)

    if LANDMARK_CHOICE == "nose":
        nose = landmarks[0]
        if not is_visible(nose):
            return None
        return (int(nose.x * frame_w), int(nose.y * frame_h))

    l_sh = landmarks[11]
    r_sh = landmarks[12]
    if not is_visible(l_sh) or not is_visible(r_sh):
        return None
    mx = int(((l_sh.x + r_sh.x) / 2) * frame_w)
    my = int(((l_sh.y + r_sh.y) / 2) * frame_h)
    return (mx, my)

def draw_skeleton(frame, landmarks: list) -> None:
    frame_h, frame_w = frame.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
        (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32), (27, 31), (28, 32)
    ]

    for start_idx, end_idx in connections:
        lm_start = landmarks[start_idx]
        lm_end = landmarks[end_idx]
        
        if getattr(lm_start, "visibility", 1.0) < 0.5 or getattr(lm_end, "visibility", 1.0) < 0.5:
            continue
            
        pt1 = (int(lm_start.x * frame_w), int(lm_start.y * frame_h))
        pt2 = (int(lm_end.x * frame_w), int(lm_end.y * frame_h))
        cv2.line(frame, pt1, pt2, (80, 80, 80), 1)
        
    for lm in landmarks:
        if getattr(lm, "visibility", 1.0) < 0.5:
            continue
        pt = (int(lm.x * frame_w), int(lm.y * frame_h))
        cv2.circle(frame, pt, 2, (190, 190, 190), -1)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    model_path = ensure_model()

    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_segmentation_masks=False,
    )

    landmarker = PoseLandmarker.create_from_options(options)
    print("[INFO] PoseLandmarker ready")

    # Using DirectShow (cv2.CAP_DSHOW) to force Windows camera compatibility
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {CAMERA_INDEX}.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab an initial frame from the webcam.")

    frame_h, frame_w = frame.shape[:2]
    
    # ── Create and scale the window ───────────────────────────────────────────
    cv2.namedWindow("Subway Surfers — Dynamic Gesture Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Subway Surfers — Dynamic Gesture Controller", 640, 480)

    # ── Pin window to the top ─────────────────────────────────────────────────
    cv2.setWindowProperty("Subway Surfers — Dynamic Gesture Controller", cv2.WND_PROP_TOPMOST, 1)
    
    # ── Runtime state ─────────────────────────────────────────────────────────
    last_press_time: float = 0.0
    last_action:     str   = ""
    awaiting_reset:  bool  = False
    anchor_pt:       list  = None    
    is_recentered:   bool  = True

    while True:
        ret, raw = cap.read()
        if not ret:
            continue

        frame = cv2.flip(raw, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        pose_found = len(result.pose_landmarks) > 0
        landmarks  = result.pose_landmarks[0] if pose_found else []

        # ── Key decision (Floating Anchor) ───────────────────────────────────
        tracking_pt = None
        in_cooldown = (time.time() - last_press_time) < COOLDOWN_SECONDS

        if pose_found:
            tracking_pt = get_tracking_point(landmarks, frame_w, frame_h)

            if tracking_pt:
                if anchor_pt is None:
                    anchor_pt = list(tracking_pt)

                # Snap instantly after a 1-second gap
                time_since_action = time.time() - last_press_time
                if time_since_action >= 0.5 and not is_recentered:
                    anchor_pt = list(tracking_pt)
                    is_recentered = True
                dx = tracking_pt[0] - anchor_pt[0]
                dy = tracking_pt[1] - anchor_pt[1]

                action = None
                if dx < -THRESHOLD_X:   action = "left"
                elif dx > THRESHOLD_X:  action = "right"
                elif dy < -THRESHOLD_Y: action = "up"
                elif dy > THRESHOLD_Y:  action = "down"

                if action is None:
                    awaiting_reset = False
                elif not awaiting_reset and not in_cooldown:
                    pyautogui.press(action)
                    last_press_time = time.time()
                    last_action     = action.upper()
                    awaiting_reset  = True
                    is_recentered   = False
                    print(f"[ACTION] → {last_action} (dx:{int(dx)}, dy:{int(dy)})")

        # ── Drawing ───────────────────────────────────────────────────────────
        if pose_found:
            draw_skeleton(frame, landmarks)

        if tracking_pt:
            cv2.circle(frame, tracking_pt, DOT_RADIUS, DOT_COLOR, -1)
            cv2.circle(frame, tracking_pt, DOT_RADIUS + 3, (255,255,255), 2)

        if anchor_pt is not None:
            ax, ay = int(anchor_pt[0]), int(anchor_pt[1])
            box_color = BOX_COLOR_COOL if in_cooldown else BOX_COLOR_IDLE
            
            cv2.drawMarker(frame, (ax, ay), box_color, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
            cv2.rectangle(frame, (ax - THRESHOLD_X, ay - THRESHOLD_Y), 
                                 (ax + THRESHOLD_X, ay + THRESHOLD_Y), box_color, 1)

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (frame_w, 42), (18, 18, 18), -1)

        elapsed  = time.time() - last_press_time
        bar_frac = min(elapsed / COOLDOWN_SECONDS, 1.0)
        bar_col  = (0, 200, 80) if bar_frac >= 1.0 else (0, 110, 230)
        cv2.rectangle(frame, (10, 8), (10 + int((frame_w - 20) * bar_frac), 34), bar_col, -1)

        status_txt = "READY" if bar_frac >= 1.0 else "COOLDOWN"
        cv2.putText(frame, status_txt, (16, 28), FONT, 0.65, (255,255,255), 2)

        if last_action:
            cv2.putText(frame, f"Last: {last_action}", (frame_w - 155, 28), FONT, 0.65, (255, 220, 50), 2)

        if not pose_found:
            cv2.putText(frame, "No pose detected — step into frame!", (frame_w // 2 - 220, frame_h // 2), FONT, 0.8, (0, 0, 255), 2)

        cv2.imshow("Subway Surfers — Dynamic Gesture Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()