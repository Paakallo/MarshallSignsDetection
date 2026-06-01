import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math

# --- Configuration ---
MODEL_PATH = "gesture_rf_model.joblib"

main_joint_idxs = [12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21]

# 2. Initialize MediaPipe Pose Lite for the Pi
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Queue to hold the coordinates of the last N frames
frame_buffer = []
predicted_gesture = "Waiting..."

print("Starting live prediction loop...")
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
        
    # Process with MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    current_angles = {}
    if results.pose_landmarks:
        landmark_list = results.pose_landmarks.landmark
        for idx, landmark in enumerate(landmark_list):
            # current_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            if idx in main_joint_idxs:
                if idx == 22 or idx == 21: continue

                next_landmark = landmark_list[idx+2]
                diff_y = next_landmark.y - landmark.y
                diff_x = next_landmark.x - landmark.x
                angle = math.degrees(math.atan(diff_y/diff_x))
                current_angles[f"{idx}-{idx+2}"] = angle

                # hand joints
                if idx == 16 or 15:

                    next_landmark = landmark_list[idx+4]
                    diff_y = next_landmark.y - landmark.y
                    diff_x = next_landmark.x - landmark.x
                    angle = math.degrees(math.atan(diff_y/diff_x))
                    current_angles[f"{idx}-{idx+4}"] = angle

    else:
        current_landmarks = [0.0] * (33 * 4) # Fallback if person is lost

    # simple logic (1 sign)
    try:
        print(f"ANGLES 12-14: {current_angles['12-14']}\n")
        print(f"ANGLES 14-16: {current_angles['14-16']}\n")
    except:
        print("Bibki")
     

        
    # Display prediction on screen
    cv2.putText(frame, predicted_gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Pi Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
