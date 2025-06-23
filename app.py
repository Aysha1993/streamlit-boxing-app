import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import time

# Load MoveNet MultiPose
@st.cache_resource
def load_movenet_multipose():
    model_url = "https://tfhub.dev/google/movenet/multipose/lightning/1"
    model = hub.load(model_url)
    return model.signatures['serving_default']

# Preprocess frame for MoveNet
def preprocess_frame(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    return img

# Extract all people keypoints
def extract_all_keypoints(output):
    people = output["output_0"].numpy()  # (1,6,56)
    return people[0]  # (6, 56)

# Draw keypoints
def draw_keypoints(frame, keypoints):
    h, w, _ = frame.shape
    for i in range(17):
        x = int(keypoints[i * 3] * w)
        y = int(keypoints[i * 3 + 1] * h)
        conf = keypoints[i * 3 + 2]
        if conf > 0.2:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    return frame

# --- Punch detection logic ---
last_punch_time = {}  # {person_id: timestamp}
PUNCH_COOLDOWN = 0.8  # seconds

def allow_punch(person_id, timestamp):
    """Cooldown check."""
    last_time = last_punch_time.get(person_id, -999)
    if timestamp - last_time > PUNCH_COOLDOWN:
        last_punch_time[person_id] = timestamp
        return True
    return False

def calculate_angle(a, b, c):
    """Calculate angle between points a-b-c in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def detect_punch(person_id, keypoints, timestamp):
    keypoints = keypoints.reshape((17, 3))
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 5, 6
    LEFT_ELBOW, RIGHT_ELBOW = 7, 8
    LEFT_WRIST, RIGHT_WRIST = 9, 10
    LEFT_HIP, RIGHT_HIP = 11, 12

    nose = keypoints[NOSE][:2]
    lw, rw = keypoints[LEFT_WRIST][:2], keypoints[RIGHT_WRIST][:2]
    le, re = keypoints[LEFT_ELBOW][:2], keypoints[RIGHT_ELBOW][:2]
    ls, rs = keypoints[LEFT_SHOULDER][:2], keypoints[RIGHT_SHOULDER][:2]
    lh, rh = keypoints[LEFT_HIP][:2], keypoints[RIGHT_HIP][:2]

    dist_lw_nose = np.linalg.norm(lw - nose)
    dist_rw_nose = np.linalg.norm(rw - nose)
    left_elbow_angle = calculate_angle(ls, le, lw)
    right_elbow_angle = calculate_angle(rs, re, rw)
    left_shoulder_angle = calculate_angle(le, ls, lh)
    right_shoulder_angle = calculate_angle(re, rs, rh)
    head_height = nose[1]

    scores = {}

    # Add score for Left Jab
    if dist_lw_nose > 100 and left_elbow_angle > 150 and np.linalg.norm(lw - le) > 40:
        scores["Left Jab"] = left_elbow_angle + dist_lw_nose

    # Add score for Right Cross
    if dist_rw_nose > 100 and right_elbow_angle > 150 and np.linalg.norm(rw - re) > 40:
        scores["Right Cross"] = right_elbow_angle + dist_rw_nose

    # Add score for Hook
    if ((left_elbow_angle < 100 and left_shoulder_angle > 80) or
        (right_elbow_angle < 100 and right_shoulder_angle > 80)):
        scores["Hook"] = 180 - min(left_elbow_angle, right_elbow_angle)

    # Add score for Duck
    if head_height > rs[1] + 40 and head_height > ls[1] + 40:
        scores["Duck"] = head_height - max(rs[1], ls[1])

    # Add score for Guard
    if dist_lw_nose < 50 and dist_rw_nose < 50:
        scores["Guard"] = 100 - (dist_lw_nose + dist_rw_nose)

    if scores:
        punch_type = max(scores, key=scores.get)
        if allow_punch(person_id, timestamp):
            return punch_type

    return "None"

# --- Streamlit App ---
st.title("🥊 MultiPose Boxing Analyzer + Punch Classifier")
uploaded_files = st.file_uploader("Upload Multiple Boxing Videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)

movenet = load_movenet_multipose()
all_logs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.info(f"Processing {uploaded_file.name}...")
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_video.name)
        video_name = uploaded_file.name
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_image = preprocess_frame(frame)
            result = movenet(input_image)
            people_keypoints = extract_all_keypoints(result)
            timestamp = time.time()

            for person_id, person_kps in enumerate(people_keypoints):
                score = person_kps[-2]
                if score < 0.3:
                    continue

                keypoints = person_kps[:51]  # 17*3 = 51
                punch_label = detect_punch(person_id, keypoints, timestamp)

                all_logs.append({
                    "Video": video_name,
                    "Frame": frame_idx,
                    "Person": person_id,
                    "Label": punch_label,
                    **{f"kp_{i}": val for i, val in enumerate(keypoints)}
                })

                draw_keypoints(frame, keypoints)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     caption=f"{video_name} - Frame {frame_idx}", use_container_width=True)

            frame_idx += 1
            if frame_idx > 10:  # demo limit
                break

        cap.release()

    # Save CSV
    df = pd.DataFrame(all_logs)
    st.subheader("📄 Combined Keypoint CSV")
    st.dataframe(df)
    csv_path = "all_punch_logs.csv"
    df.to_csv(csv_path, index=False)
    st.download_button("Download CSV", df.to_csv(index=False), file_name="all_punch_logs.csv")

    # Train Classifier
    st.subheader("🧠 Train RandomForest Classifier")
    if st.button("Train on All Videos"):
        if df.shape[0] > 5:
            feature_cols = [col for col in df.columns if col.startswith("kp_")]
            X = df[feature_cols].fillna(0)
            y = df["Label"]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y_encoded)
            joblib.dump(model, "punch_classifier.pkl")
            st.success("✅ Model trained and saved as punch_classifier.pkl")
            y_pred = model.predict(X)
            st.text("Classification Report:")
            st.text(classification_report(y_encoded, y_pred, target_names=le.classes_))
        else:
            st.warning("Need more than 5 samples to train.")

# ✅ Requirements
with open("requirements.txt", "w") as f:
    f.write('''streamlit
tensorflow
tensorflow_hub
opencv-python-headless
pandas
numpy
scikit-learn
joblib
ffmpeg-python
''')
print("✅ requirements.txt saved")
