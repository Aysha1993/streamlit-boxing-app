import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from sklearn import svm
from joblib import dump

st.set_page_config(page_title="Boxing Analyzer", layout="wide")
st.title("🥊 Boxing Analyzer with MoveNet MultiPose")

# Load MoveNet MultiPose
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

model = load_model()

# Draw pose skeleton
def draw_skeleton(frame, keypoints):
    for person in keypoints:
        for i, kp in enumerate(person):
            if kp[2] > 0.2:
                x, y = int(kp[1] * frame.shape[1]), int(kp[0] * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    return frame

# Extract keypoints from MoveNet MultiPose output
def extract_keypoints(results):
    keypoints_with_scores = results['output_0'].numpy()
    keypoints = []
    for person in keypoints_with_scores[0]:
        score = person[55]  # overall confidence
        if score < 0.2:
            continue
        person_kps = []
        for i in range(17):
            y = person[i * 3]
            x = person[i * 3 + 1]
            conf = person[i * 3 + 2]
            person_kps.append([y, x, conf])
        keypoints.append(person_kps)
    return keypoints

# Punch classification
def classify_punch(keypoints):
    punch_type = []
    for person in keypoints:
        if len(person) != 17:
            punch_type.append("unknown")
            continue
        lwrist = person[9]
        rwrist = person[10]
        lshoulder = person[5]
        rshoulder = person[6]
        lelbow = person[7]
        relbow = person[8]

        if lwrist[2] > 0.2 and lshoulder[2] > 0.2 and lwrist[0] < lshoulder[0]:
            punch_type.append("jab")
        elif rwrist[2] > 0.2 and rshoulder[2] > 0.2 and rwrist[0] < rshoulder[0]:
            punch_type.append("cross")
        elif lelbow[2] > 0.2 and abs(lelbow[1] - lwrist[1]) > 0.1:
            punch_type.append("hook")
        else:
            punch_type.append("guard")
    return punch_type

# Posture analysis
def check_posture(keypoints):
    feedback = []
    for person in keypoints:
        if len(person) != 17:
            feedback.append("unknown")
            continue
        lelbow = person[7]
        relbow = person[8]
        lhip = person[11]
        rhip = person[12]

        elbow_drop = lelbow[0] > lhip[0] and relbow[0] > rhip[0]
        feedback.append("Elbow drop detected" if elbow_drop else "Good posture")
    return feedback

# Glove detection (presence of wrists)
def detect_gloves(keypoints):
    gloves = []
    for person in keypoints:
        lwrist = person[9]
        rwrist = person[10]
        gloves.append(f"Left: {'yes' if lwrist[2] > 0.2 else 'no'}, Right: {'yes' if rwrist[2] > 0.2 else 'no'}")
    return gloves

# File uploader
uploaded_files = st.file_uploader("📂 Upload MP4 boxing video(s)", type=["mp4"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"🎞️ {uploaded_file.name}")
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_file.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        base_name = os.path.splitext(uploaded_file.name)[0]
        out_path = f"/tmp/{base_name}_out.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        punch_log = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (256, 256))
            img = tf.convert_to_tensor(resized, dtype=tf.uint8)
            input_tensor = tf.expand_dims(img, axis=0)
            input_tensor = tf.cast(input_tensor, dtype=tf.int32)

            results = model.signatures['serving_default'](input_tensor)
            keypoints = extract_keypoints(results)

            frame = draw_skeleton(frame, keypoints)
            punches = classify_punch(keypoints)
            postures = check_posture(keypoints)
            gloves = detect_gloves(keypoints)

            for i, punch in enumerate(punches):
                punch_log.append({
                    "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "person": i,
                    "punch": punch,
                    "posture": postures[i],
                    "gloves": gloves[i]
                })

            out.write(frame)

        cap.release()
        out.release()

        # Display output video
        st.video(out_path)
        st.success("✅ Video processed!")

        # Create DataFrame
        df = pd.DataFrame(punch_log)
        st.dataframe(df)

        csv_path = f"/tmp/{base_name}_punch_log.csv"
        model_path = f"/tmp/{base_name}_svm_model.joblib"
        df.to_csv(csv_path, index=False)

        # Download CSV button
        with open(csv_path, "rb") as f:
            st.download_button("⬇️ Download Punch Log CSV", f, file_name=f"{base_name}_punch_log.csv")

        # Train SVM
        if st.button(f"🧠 Train SVM Model on {uploaded_file.name}"):
            if 'punch' in df.columns:
                X = df[['frame', 'person']]
                y = df['punch']
                clf = svm.SVC()
                clf.fit(X, y)
                dump(clf, model_path)
                st.success("✅ SVM model trained and saved!")
