import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
from sklearn import svm
from joblib import dump
import io

st.set_page_config(layout="wide")
st.title("🥊 Fast Boxing Analyzer")

model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

def draw_skeleton(frame, keypoints):
    for person in keypoints:
        for kp in person:
            if kp[2] > 0.2:
                x, y = int(kp[1] * frame.shape[1]), int(kp[0] * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    return frame

def extract_keypoints(results):
    keypoints = []
    for person in results['output_0'][0]:
        person_kps = []
        for i in range(17):
            y, x, score = person[i*3:(i+1)*3]
            person_kps.append([y, x, score])
        keypoints.append(person_kps)
    return keypoints

def classify_punch(keypoints):
    punches = []
    for person in keypoints:
        if len(person) != 17:
            punches.append("unknown")
            continue
        lwrist, rwrist = person[9], person[10]
        lshoulder, rshoulder = person[5], person[6]
        lelbow = person[7]

        if lwrist[2] > 0.2 and lshoulder[2] > 0.2 and lwrist[0] < lshoulder[0]:
            punches.append("jab")
        elif rwrist[2] > 0.2 and rshoulder[2] > 0.2 and rwrist[0] < rshoulder[0]:
            punches.append("cross")
        elif lelbow[2] > 0.2 and abs(lelbow[1] - lwrist[1]) > 0.1:
            punches.append("hook")
        else:
            punches.append("guard")
    return punches

def check_posture(keypoints):
    feedback = []
    for person in keypoints:
        if len(person) != 17:
            feedback.append("unknown")
            continue
        lelbow, relbow = person[7], person[8]
        lhip, rhip = person[11], person[12]
        elbow_drop = lelbow[0] > lhip[0] and relbow[0] > rhip[0]
        feedback.append("Elbow drop" if elbow_drop else "Good posture")
    return feedback

def detect_gloves(keypoints):
    gloves = []
    for person in keypoints:
        lwrist, rwrist = person[9], person[10]
        gloves.append(f"L: {'yes' if lwrist[2] > 0.2 else 'no'}, R: {'yes' if rwrist[2] > 0.2 else 'no'}")
    return gloves

uploaded_file = st.file_uploader("Upload a short MP4 video (compressed)", type=["mp4"])
print("uploaded_file")

if uploaded_file:
    st.subheader("🎬 Preview and Analyze")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    with st.spinner("🔍 Processing video..."):
        cap = cv2.VideoCapture(tfile.name)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

        punch_log = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (256, 256))
            input_tensor = tf.convert_to_tensor(resized, dtype=tf.uint8)
            input_tensor = tf.expand_dims(input_tensor, 0)
            input_tensor = tf.cast(input_tensor, dtype=tf.int32)

            results = model.signatures['serving_default'](input_tensor)
            keypoints = extract_keypoints(results)

            frame = draw_skeleton(frame, keypoints)
            punches = classify_punch(keypoints)
            postures = check_posture(keypoints)
            gloves = detect_gloves(keypoints)

            for i, punch in enumerate(punches):
                punch_log.append({
                    "frame": cap.get(cv2.CAP_PROP_POS_FRAMES),
                    "person": i,
                    "punch": punch,
                    "posture": postures[i],
                    "gloves": gloves[i]
                })

            out.write(frame)

        cap.release()
        out.release()

    st.success("✅ Video processed!")

    with open(temp_output.name, 'rb') as f:
        st.video(f.read())

    df = pd.DataFrame(punch_log)
    st.dataframe(df)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download CSV Log", csv_buf.getvalue(), "punch_log.csv", "text/csv")

    if st.button("Train Punch Classifier"):
        if 'punch' in df.columns:
            X = df[['frame', 'person']]
            y = df['punch']
            clf = svm.SVC()
            clf.fit(X, y)
            model_buf = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
            dump(clf, model_buf.name)
            st.success("✅ SVM model trained.")
            with open(model_buf.name, "rb") as f:
                st.download_button("⬇️ Download SVM Model", f, "svm_model.joblib")
