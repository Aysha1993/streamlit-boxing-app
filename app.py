import streamlit as st
import av
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import pandas as pd
from collections import deque, Counter
from io import BytesIO
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle

# ------------------------------
# Constants and utils functions
# ------------------------------

KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

POSE_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)
]

PUNCH_CLASSES = ['jab_left', 'jab_right', 'cross_left', 'cross_right', 'hook_left', 'hook_right', 'none']

MODEL_SAVE_PATH = 'dummy_punch_model.pkl'

def draw_skeleton(image, keypoints, confidence_threshold=0.3):
    h, w, _ = image.shape
    for idx1, idx2 in POSE_CONNECTIONS:
        y1, x1, c1 = keypoints[idx1]
        y2, x2, c2 = keypoints[idx2]
        if c1 > confidence_threshold and c2 > confidence_threshold:
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(image, pt1, pt2, (0,255,0), 2)
    for i, (y, x, c) in enumerate(keypoints):
        if c > confidence_threshold:
            cv2.circle(image, (int(x * w), int(y * h)), 4, (0,0,255), -1)
    return image

def detect_gloves(keypoints, threshold=0.2):
    # Dummy glove detection: If wrist confidence high & wrist above hip (simple proxy), detect glove
    left_wrist = keypoints[KEYPOINT_DICT['left_wrist']]
    right_wrist = keypoints[KEYPOINT_DICT['right_wrist']]
    left_hip = keypoints[KEYPOINT_DICT['left_hip']]
    right_hip = keypoints[KEYPOINT_DICT['right_hip']]

    gloves = {'left_glove': False, 'right_glove': False}

    if left_wrist[2] > threshold and left_wrist[0] < left_hip[0]:
        gloves['left_glove'] = True
    if right_wrist[2] > threshold and right_wrist[0] < right_hip[0]:
        gloves['right_glove'] = True

    return gloves

def calculate_angle(a, b, c):
    """Calculate angle (degrees) at point b from points a-b-c"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def posture_labels_and_corrections(keypoints):
    # Simple rules: check elbow, knee angles, shoulder alignment, wrist angles
    labels = {}
    corrections = {}

    left_shoulder = keypoints[KEYPOINT_DICT['left_shoulder']][:2]
    left_elbow = keypoints[KEYPOINT_DICT['left_elbow']][:2]
    left_wrist = keypoints[KEYPOINT_DICT['left_wrist']][:2]
    left_hip = keypoints[KEYPOINT_DICT['left_hip']][:2]
    left_knee = keypoints[KEYPOINT_DICT['left_knee']][:2]
    left_ankle = keypoints[KEYPOINT_DICT['left_ankle']][:2]

    right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']][:2]
    right_elbow = keypoints[KEYPOINT_DICT['right_elbow']][:2]
    right_wrist = keypoints[KEYPOINT_DICT['right_wrist']][:2]
    right_hip = keypoints[KEYPOINT_DICT['right_hip']][:2]
    right_knee = keypoints[KEYPOINT_DICT['right_knee']][:2]
    right_ankle = keypoints[KEYPOINT_DICT['right_ankle']][:2]

    # Angles
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    labels['left_elbow_angle'] = left_elbow_angle
    labels['right_elbow_angle'] = right_elbow_angle
    labels['left_knee_angle'] = left_knee_angle
    labels['right_knee_angle'] = right_knee_angle

    # Corrections
    corrections['left_elbow_drop'] = left_elbow_angle < 90  # example rule: elbow should not be less than 90°
    corrections['right_elbow_drop'] = right_elbow_angle < 90
    corrections['left_knee_straight'] = left_knee_angle > 160
    corrections['right_knee_straight'] = right_knee_angle > 160

    return labels, corrections

def dummy_punch_classifier(keypoints):
    # Dummy heuristic: compare x positions of wrists and elbows for punch type
    # Just randomly assign punch types to show concept
    # A real model would use features & ML
    left_wrist = keypoints[KEYPOINT_DICT['left_wrist']]
    right_wrist = keypoints[KEYPOINT_DICT['right_wrist']]
    left_elbow = keypoints[KEYPOINT_DICT['left_elbow']]
    right_elbow = keypoints[KEYPOINT_DICT['right_elbow']]

    # If wrists have low confidence, no punch
    if left_wrist[2] < 0.3 and right_wrist[2] < 0.3:
        return 'none'

    # Dummy rules
    if right_wrist[0] > right_elbow[0] + 0.05:
        return 'jab_right'
    if right_wrist[0] < right_elbow[0] - 0.05:
        return 'cross_right'
    if left_wrist[0] < left_elbow[0] - 0.05:
        return 'jab_left'
    if left_wrist[0] > left_elbow[0] + 0.05:
        return 'cross_left'

    return 'none'

def process_video_frames(container, model):
    frames_data = []
    pose_results = []
    punch_preds = []
    punch_start = False
    punch_events = []

    # For speed approx: track wrist x-pos over frames
    wrist_history = deque(maxlen=5)

    for frame_idx, frame in enumerate(container.decode(video=0)):
        img = frame.to_ndarray(format='rgb24')
        img_resized = tf.image.resize(img, (256,256))
        input_tensor = tf.expand_dims(img_resized, axis=0)
        input_tensor = tf.cast(input_tensor, dtype=tf.int32)

        # Pose detection
        outputs = model(input_tensor)
        keypoints_with_scores = outputs['output_0'].numpy()
        # shape: [1, 6, 17, 3] -> batch, 6 poses, 17 keypoints, (y,x,score)
        pose = keypoints_with_scores[0][0]  # take first pose

        # Punch classification
        punch = dummy_punch_classifier(pose)

        # Detect glove
        gloves = detect_gloves(pose)

        # Posture labels and corrections
        labels, corrections = posture_labels_and_corrections(pose)

        # Track punch start/end
        if punch != 'none' and not punch_start:
            punch_start = True
            punch_events.append({'frame': frame_idx, 'event': 'punch_start', 'punch': punch})
        if punch == 'none' and punch_start:
            punch_start = False
            punch_events.append({'frame': frame_idx, 'event': 'punch_end', 'punch': punch})

        # Speed approx: use difference in right wrist x over last frames
        wrist_x = pose[KEYPOINT_DICT['right_wrist']][1]
        wrist_history.append(wrist_x)
        speed = 0
        if len(wrist_history) >= 2:
            speed = np.abs(wrist_history[-1] - wrist_history[-2]) * 100  # arbitrary scale

        frames_data.append({
            'frame_idx': frame_idx,
            'punch': punch,
            'gloves': gloves,
            'labels': labels,
            'corrections': corrections,
            'speed': speed
        })
        pose_results.append(pose)
        punch_preds.append(punch)

    return frames_data, pose_results, punch_preds, punch_events

# ------------------------------
# Streamlit App Start
# ------------------------------

st.set_page_config(page_title="Boxing Pose & Punch Analyzer", layout="wide")

st.title("Boxing Analyzer with MoveNet MultiPose + Punch Detection")

# Load MoveNet MultiPose model from TFHub (cached)
@st.cache_resource
def load_movenet():
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    return model

model = load_movenet()

uploaded_video = st.file_uploader("Upload Boxing Video (mp4, mov, avi)", type=['mp4', 'mov', 'avi'])

if uploaded_video is not None:
    with st.spinner("Decoding video and running pose detection..."):
        # Save video temporarily for PyAV
        temp_video = 'temp_input_video.mp4'
        with open(temp_video, 'wb') as f:
            f.write(uploaded_video.getbuffer())

        # Open video container using PyAV
        container = av.open(temp_video)

        # Process video frames: pose detection, punch classification, glove detection, posture
        frames_data, pose_results, punch_preds, punch_events = process_video_frames(container, model)

        st.success(f"Processed {len(frames_data)} frames.")

        # Display annotated video preview with skeleton + punches (simplified)
        st.header("Annotated Frames Preview")
        for i in range(0, len(frames_data), max(1, len(frames_data)//20)):  # Sample 20 frames for preview
            pose = pose_results[i]
            img = np.zeros((256,256,3), dtype=np.uint8) + 255
            img = draw_skeleton(img, pose)
            gloves = frames_data[i]['gloves']
            punch = frames_data[i]['punch']
            label = f"Punch: {punch} | Gloves: {gloves}"
            cv2.putText(img, label, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            st.image(img, caption=f"Frame {i}")

        # Punch count summary
        punch_counts = Counter([p for p in punch_preds if p != 'none'])
        st.subheader("Punch Counts")
        st.write(punch_counts)

        # Punch speed histogram
        speeds = [f['speed'] for f in frames_data if f['punch'] != 'none']
        st.subheader("Punch Speed Approximation")
        if speeds:
            fig, ax = plt.subplots()
            ax.hist(speeds, bins=10, color='orange')
            ax.set_xlabel('Speed')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        else:
            st.write("No punches detected to calculate speed.")

        # Dummy model training / saving / loading
        st.header("Dummy Punch Classifier Model Training & Evaluation")

        # Prepare dummy feature matrix X and labels y (from keypoints)
        # Here we use the right_wrist x coordinate & left_wrist x coordinate as dummy features
        X = []
        y = []
        for d in frames_data:
            pose = d['labels']
            punch = d['punch']
            # Use wrist angles as features (or positions)
            left_wrist_x = pose_results[d['frame_idx']][KEYPOINT_DICT['left_wrist']][1]
            right_wrist_x = pose_results[d['frame_idx']][KEYPOINT_DICT['right_wrist']][1]
            X.append([left_wrist_x, right_wrist_x])
            y.append(punch)

        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(X_train, y_train)

        y_pred = dummy_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.write(f"Dummy Classifier Accuracy (Most Frequent): {accuracy:.2f}")

        # Save model
        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(dummy_clf, f)
        st.write(f"Dummy model saved as {MODEL_SAVE_PATH}")

        # Load model and predict again for demo
        with open(MODEL_SAVE_PATH, 'rb') as f:
            loaded_clf = pickle.load(f)
        y_pred_loaded = loaded_clf.predict(X_test)
        st.write("Loaded model predictions on test data:")
        st.write(y_pred_loaded[:10])

        # Confusion matrix plot
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_loaded, labels=PUNCH_CLASSES)
        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=PUNCH_CLASSES, yticklabels=PUNCH_CLASSES, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

        # Punch distribution pie chart
        st.subheader("Punch Distribution Pie Chart")
        punch_dist = Counter(y)
        fig2, ax2 = plt.subplots()
        ax2.pie(punch_dist.values(), labels=punch_dist.keys(), autopct='%1.1f%%')
        st.pyplot(fig2)

        # Posture correction bar chart summary
        correction_counts = Counter()
        for d in frames_data:
            for k,v in d['corrections'].items():
                if v:
                    correction_counts[k] += 1
        st.subheader("Posture Corrections Frequency")
        fig3, ax3 = plt.subplots()
        ax3.bar(correction_counts.keys(), correction_counts.values())
        ax3.set_ylabel('Counts')
        ax3.set_xticklabels(correction_counts.keys(), rotation=45, ha='right')
        st.pyplot(fig3)

        # Save punch & posture data as CSV
        st.header("Download Punch & Posture Data as CSV")

        df = pd.DataFrame([{
            'frame_idx': d['frame_idx'],
            'punch': d['punch'],
            'left_glove': d['gloves']['left_glove'],
            'right_glove': d['gloves']['right_glove'],
            'speed': d['speed'],
            **{k: v for k,v in d['labels'].items()},
            **{k: v for k,v in d['corrections'].items()}
        } for d in frames_data])

        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(label="Download CSV", data=csv_bytes, file_name="punch_posture_data.csv", mime='text/csv')

        # Cleanup temp file
        if os.path.exists(temp_video):
            os.remove(temp_video)

else:
    st.info("Upload a boxing video to analyze punches, poses, and posture.")


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
matplotlib
'''

with open("requirements.txt", "w") as f:
    f.write(requirements)
print("✅ requirements.txt saved")
