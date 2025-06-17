import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import tempfile


# Load MoveNet MultiPose
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']

# Constants
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# HSV ranges for red and blue jerseys
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# Punch thresholds
PUNCH_DISTANCE_THRESHOLD = 50  # pixel movement of wrist



def detect_pose(image):
    input_img = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
    outputs = movenet(input_img)

    # Shape: (1, N, 56), where N = number of detected persons
    keypoints_all = outputs['output_0'].numpy()
    num_persons = keypoints_all.shape[1]

    # (1, N, 56) → (N, 17, 3): first 51 values are 17 keypoints * 3 (y, x, score)
    keypoints = keypoints_all[0, :, :51].reshape((num_persons, 17, 3))
    return keypoints


# Crop upper body to detect jersey color
def get_torso_patch(frame, keypoints):
    h, w, _ = frame.shape
    left_shoulder = keypoints[5][:2] * [w, h]
    right_shoulder = keypoints[6][:2] * [w, h]
    left_hip = keypoints[11][:2] * [w, h]
    right_hip = keypoints[12][:2] * [w, h]

    x1 = int(min(left_shoulder[0], right_shoulder[0]))
    x2 = int(max(left_shoulder[0], right_shoulder[0]))
    y1 = int(min(left_shoulder[1], right_shoulder[1]))
    y2 = int(max(left_hip[1], right_hip[1]))

    return frame[y1:y2, x1:x2]

# Detect jersey color
def detect_color(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    red_score = cv2.countNonZero(red_mask)
    blue_score = cv2.countNonZero(blue_mask)
    if red_score > blue_score and red_score > 200:
        return 'red'
    elif blue_score > red_score and blue_score > 200:
        return 'blue'
    return 'unknown'

# Punch detection by wrist movement
def detect_punch(wrist_history, new_point):
    if len(wrist_history) < 1:
        wrist_history.append(new_point)
        return False
    last = wrist_history[-1]
    movement = np.linalg.norm(np.array(new_point) - np.array(last))
    wrist_history.append(new_point)
    return movement > PUNCH_DISTANCE_THRESHOLD


uploaded_file = st.file_uploader("Upload a boxing video", type=["mp4", "mov", "avi"])
# Main processing
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    red_punches = 0
    blue_punches = 0
    red_history = []
    blue_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = detect_pose(frame)[0:6]  # up to 6 persons
        h, w, _ = frame.shape

        for person in keypoints:
            if person[0][2] < 0.2:
                continue
            kps = person[:17]
            torso_patch = get_torso_patch(frame, kps)
            if torso_patch.size == 0:
                continue
            jersey_color = detect_color(torso_patch)

            left_wrist = kps[9][:2] * [w, h]
            right_wrist = kps[10][:2] * [w, h]

            wrist_center = ((left_wrist[0]+right_wrist[0])/2, (left_wrist[1]+right_wrist[1])/2)

            if jersey_color == 'red':
                if detect_punch(red_history, wrist_center):
                    red_punches += 1
                    cv2.putText(frame, "Red Punch!", (int(wrist_center[0]), int(wrist_center[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(frame, (int(wrist_center[0]), int(wrist_center[1])), 6, (0, 0, 255), -1)
            elif jersey_color == 'blue':
                if detect_punch(blue_history, wrist_center):
                    blue_punches += 1
                    cv2.putText(frame, "Blue Punch!", (int(wrist_center[0]), int(wrist_center[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.circle(frame, (int(wrist_center[0]), int(wrist_center[1])), 6, (255, 0, 0), -1)

        # Show counter
        cv2.putText(frame, f"Red Punches: {red_punches}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Blue Punches: {blue_punches}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        

    cap.release()
    cv2.destroyAllWindows()

if uploaded_file is not None:
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

    if st.button("Start Processing"):
        process_video(video_path)


requirements = '''streamlit
tensorflow
tensorflow_hub
opencv-python-headless
pandas
numpy
scikit-learn
joblib
ffmpeg-python
tqdm
seaborn
lightgbm
imbalanced-learn
plotly
matplotlib
'''

with open("requirements.txt", "w") as f:
    f.write(requirements)
print("✅ requirements.txt saved")
