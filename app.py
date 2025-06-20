import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load MoveNet MultiPose
@st.cache_resource
def load_movenet_multipose():
    model_url = "https://tfhub.dev/google/movenet/multipose/lightning/1"
    model = hub.load(model_url)
    return model.signatures['serving_default']

# Preprocess for MoveNet MultiPose
def preprocess_frame(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    return img

# Extract keypoints for all detected persons
def extract_all_keypoints(output):
    people = output["output_0"].numpy()  # shape: (1,6,56)
    return people[0]  # (6, 56) = 17 keypoints * 3 (x,y,conf) + score + id

# Draw keypoints on frame
def draw_keypoints(frame, keypoints):
    h, w, _ = frame.shape
    for i in range(17):
        x = int(keypoints[i * 3] * w)
        y = int(keypoints[i * 3 + 1] * h)
        conf = keypoints[i * 3 + 2]
        if conf > 0.2:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    return frame

# Constants
all_logs = []
st.title("🥊 Multi-Video Boxing Analyzer + Trainer (MultiPose)")

uploaded_files = st.file_uploader("Upload Multiple Boxing Videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)
movenet = load_movenet_multipose()

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

            for person_id, person_kps in enumerate(people_keypoints):
                score = person_kps[-2]
                if score < 0.3:
                    continue

                keypoints = person_kps[:51]  # x,y,conf * 17
                label = st.selectbox(
                    f"{video_name} - Frame {frame_idx} - Person {person_id}",
                    ["Jab", "Cross", "Hook", "None"],
                    key=f"{video_name}_{frame_idx}_{person_id}"
                )

                all_logs.append({
                    "Video": video_name,
                    "Frame": frame_idx,
                    "Person": person_id,
                    "Label": label,
                    **{f"kp_{i}": val for i, val in enumerate(keypoints)}
                })

                draw_keypoints(frame, keypoints)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     caption=f"{video_name} - Frame {frame_idx}", use_column_width=True)

            frame_idx += 1
            if frame_idx > 10:
                break

        cap.release()

    # Save combined CSV
    df = pd.DataFrame(all_logs)
    st.subheader("📄 Combined Keypoint CSV")
    st.dataframe(df)

    csv_path = "/content/all_punch_logs.csv"
    df.to_csv(csv_path, index=False)
    st.download_button("Download Combined CSV", df.to_csv(index=False), file_name="all_punch_logs.csv")

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

            joblib.dump(model, "/content/punch_classifier.pkl")
            st.success("✅ Model trained and saved as punch_classifier.pkl")

            y_pred = model.predict(X)
            st.text("Classification Report:")
            st.text(classification_report(y_encoded, y_pred, target_names=le.classes_))

        else:
            st.warning("Need more than 5 labeled frames to train.")

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
