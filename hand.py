from flask import Flask, render_template, request
import threading
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os 
from collections import deque

app = Flask(__name__)

# Configuration
os.environ['OMP_NUM_THREADS'] = '1'
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.02  # Slightly increased for stability

# Global control
hand_tracking_active = False
tracking_thread = None

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Smoothing
cursor_positions = deque(maxlen=5)

# Click control
last_click_time = 0
click_delay = 0.3  # Seconds between clicks

def apply_moving_average(new_position):
    cursor_positions.append(new_position)
    return np.mean(cursor_positions, axis=0).astype(int)

def hand_tracking():
    global hand_tracking_active, last_click_time
    
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        # Camera setup
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        screen_width, screen_height = pyautogui.size()
        
        # State tracking
        dragging = False
        left_click_active = False
        right_click_active = False
        
        while hand_tracking_active and cap.isOpened():
            success, img = cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    width, height = img.shape[1], img.shape[0]

                    # Get landmarks
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    # Convert to pixel coordinates
                    index_pos = np.array([index_tip.x * width, index_tip.y * height])
                    thumb_pos = np.array([thumb_tip.x * width, thumb_tip.y * height])
                    middle_pos = np.array([middle_tip.x * width, middle_tip.y * height])

                    # Calculate distances
                    dist_thumb_index = np.linalg.norm(index_pos - thumb_pos)
                    dist_thumb_middle = np.linalg.norm(middle_pos - thumb_pos)
                    dist_index_middle = np.linalg.norm(index_pos - middle_pos)

                    # Smooth cursor position
                    smoothed_pos = apply_moving_average(index_pos)
                    scaled_x = np.interp(smoothed_pos[0], [0, width], [0, screen_width])
                    scaled_y = np.interp(smoothed_pos[1], [0, height], [0, screen_height])

                    # 1. CURSOR MOVEMENT (Always active)
                    if not dragging:
                        pyautogui.moveTo(scaled_x, scaled_y)

                    # 2. LEFT CLICK (Index + Thumb pinch)
                    if dist_thumb_index < 30 and dist_index_middle > 40:
                        current_time = time.time()
                        if not left_click_active and current_time - last_click_time > click_delay:
                            pyautogui.click()
                            left_click_active = True
                            last_click_time = current_time
                            cv2.putText(img, "LEFT CLICK", (50, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        left_click_active = False

                    # 3. RIGHT CLICK (Middle + Thumb pinch)
                    if dist_thumb_middle < 30 and dist_thumb_index > 40:
                        current_time = time.time()
                        if not right_click_active and current_time - last_click_time > click_delay:
                            pyautogui.rightClick()
                            right_click_active = True
                            last_click_time = current_time
                            cv2.putText(img, "RIGHT CLICK", (50, 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    else:
                        right_click_active = False

                    # 4. DRAG AND DROP (Index + Thumb hold and move)
                    if dist_thumb_index < 30:
                        if not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                            cv2.putText(img, "DRAGGING", (50, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            pyautogui.moveTo(scaled_x, scaled_y)
                    else:
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False

                    # Visual feedback
                    cv2.circle(img, tuple(index_pos.astype(int)), 10, (0, 255, 255), 2)  # Yellow
                    if dist_thumb_index < 40:
                        cv2.circle(img, tuple(thumb_pos.astype(int)), 10, (0, 255, 0), 2)  # Green
                    if dist_thumb_middle < 40:
                        cv2.circle(img, tuple(middle_pos.astype(int)), 10, (255, 0, 0), 2)  # Blue

            cv2.imshow("Hand Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup
        if dragging:
            pyautogui.mouseUp()
        cap.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_hand_tracking', methods=['POST'])
def start_hand_tracking():
    global hand_tracking_active, tracking_thread
    hand_tracking_active = True
    if not tracking_thread or not tracking_thread.is_alive():
        tracking_thread = threading.Thread(target=hand_tracking)
        tracking_thread.daemon = True
        tracking_thread.start()
    return '', 204

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global hand_tracking_active
    hand_tracking_active = False
    time.sleep(0.5)
    return '', 204

if __name__ == '__main__':
    print("Starting Flask app at http://127.0.0.1:5000")
    app.run(debug=True)