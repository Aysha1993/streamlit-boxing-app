import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
import shutil
from sklearn import svm
from joblib import dump
import io

st.set_option('client.showErrorDetails', True)
st.title("🥊 Boxing Analyzer with Punches, Posture & Gloves")

# Load MoveNet MultiPose model
@st.cache_resource
def load_model():
    os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'
    return hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
model = load_model()

# ✅ Fix: Keypoint extraction
def extract_keypoints(results):
    people = []
    raw = results['output_0'].numpy()[0]  # shape (6, 56)
    for person_data in raw:
        keypoints = np.array(person_data[:51]).reshape(17, 3)  # y, x, confidence
        score = person_data[55]  # person score
        if score > 0.2 and np.mean(keypoints[:, 2]) > 0.2:
            people.append(keypoints.tolist())
    return people

# Punch classification
def classify_punch(keypoints):
    result = []
    for kp in keypoints:
        lw, rw = kp[9], kp[10]
        ls, rs = kp[5], kp[6]
        le, re = kp[7], kp[8]
        if lw[2] > 0.2 and ls[2] > 0.2 and lw[0] < ls[0]:
            result.append("Left Jab")
        elif rw[2] > 0.2 and rs[2] > 0.2 and rw[0] < rs[0]:
            result.append("Right Cross")
        elif le[2] > 0.2 and abs(le[1] - lw[1]) > 0.1:
            result.append("Left Hook")
        elif re[2] > 0.2 and abs(re[1] - rw[1]) > 0.1:
            result.append("Right Hook")
        else:
            result.append("Guard")
    return result

# Posture analysis
def check_posture(keypoints):
    feedback = []
    for kp in keypoints:
        msgs = []
        if kp[7][0] > kp[11][0]: msgs.append("Left Elbow ↓")
        if kp[8][0] > kp[12][0]: msgs.append("Right Elbow ↓")
        if kp[5][0] > kp[11][0]: msgs.append("Left Shoulder ↓")
        if kp[6][0] > kp[12][0]: msgs.append("Right Shoulder ↓")
        if kp[15][0] < kp[13][0] - 0.05: msgs.append("Left Knee Bent")
        if kp[16][0] < kp[14][0] - 0.05: msgs.append("Right Knee Bent")
        if kp[9][0] > kp[7][0]: msgs.append("Left Wrist ↓")
        if kp[10][0] > kp[8][0]: msgs.append("Right Wrist ↓")
        feedback.append(", ".join(msgs) if msgs else "Good Posture")
    return feedback

# Glove detection
def detect_gloves(keypoints):
    gloves = []
    for kp in keypoints:
        lw, rw = kp[9], kp[10]
        gloves.append(f"Gloves: L-{'yes' if lw[2]>0.2 else 'no'} R-{'yes' if rw[2]>0.2 else 'no'}")
    return gloves

# Draw keypoints and annotations
def draw_annotations(frame, keypoints, punches, postures, gloves):
    for i, kp in enumerate(keypoints):
        for j, (y, x, s) in enumerate(kp):
            if s > 0.2:
                cx, cy = int(x * frame.shape[1]), int(y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        label = f"{punches[i]}, {postures[i]}, {gloves[i]}"
        base_x = int(kp[0][1] * frame.shape[1])
        base_y = int(kp[0][0] * frame.shape[0]) - 20
        cv2.putText(frame, label, (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

# UI to upload videos
uploaded_files = st.file_uploader("Upload boxing MP4 videos", type=["mp4"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")

        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_path, 'wb') as out_file:
            shutil.copyfileobj(uploaded_file, out_file)

        cap = cv2.VideoCapture(input_path)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = os.path.join(temp_dir, f"output_{uploaded_file.name}")
        out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        punch_log = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (256, 256))
            input_tensor = tf.convert_to_tensor(resized[None, ...], dtype=tf.int32)
            results = model.signatures['serving_default'](input_tensor)
            keypoints = extract_keypoints(results)

            if not keypoints:
                out_writer.write(frame)
                continue

            punches = classify_punch(keypoints)
            postures = check_posture(keypoints)
            gloves = detect_gloves(keypoints)

            annotated = draw_annotations(frame.copy(), keypoints, punches, postures, gloves)
            out_writer.write(annotated)

            for i in range(len(punches)):
                punch_log.append({
                    "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "person": i,
                    "punch": punches[i],
                    "posture": postures[i],
                    "gloves": gloves[i]
                })

        cap.release()
        out_writer.release()

        st.video(output_path)
        st.success("✅ Annotated video ready")

        df = pd.DataFrame(punch_log)
        st.dataframe(df)

        # Download button
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("📥 Download CSV", csv_buffer.getvalue(), file_name=f"{uploaded_file.name}_log.csv", mime="text/csv")

        # Get base name without extension for saving the model
        base_name = os.path.splitext(uploaded_file.name)[0]  # Get the file name without extension
        model_dest = f"/tmp/{base_name}_svm_model.joblib"
        
        model_dest = f"/tmp/{base_name}_svm_model.joblib"
        if st.button(f"Train SVM on {uploaded_file.name}"):
            if 'punch' in df.columns:
                X = df[['frame', 'person']]
                y = df['punch']
                clf = svm.SVC()
                clf.fit(X, y)
                dump(clf, model_dest)
                st.success("SVM trained and saved ✅")

