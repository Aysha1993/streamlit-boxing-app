import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
import io
from sklearn import svm
from joblib import dump

# Load MoveNet Multipose Model
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    return model.signatures['serving_default']

# Detect poses
def detect_poses(frame, model):
    input_size = 256
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = model(input_img)
    keypoints = outputs['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
    return [kp.tolist() for kp in keypoints if np.mean(np.array(kp)[:, 2]) > 0.2]

# Filter top 2 confident persons
def filter_top_two_persons(keypoints):
    scored = [(np.mean([s for (_, _, s) in kp]), idx) for idx, kp in enumerate(keypoints)]
    top_two = sorted(scored, reverse=True)[:2]
    return [keypoints[i] for (_, i) in top_two]

# Draw skeleton
def draw_skeleton(frame, keypoints):
    h, w, _ = frame.shape
    edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(9,10),
             (11,12),(11,13),(13,15),(12,14),(14,16)]
    for person in keypoints:
        for p1, p2 in edges:
            if person[p1][2] > 0.2 and person[p2][2] > 0.2:
                x1, y1 = int(person[p1][1]*w), int(person[p1][0]*h)
                x2, y2 = int(person[p2][1]*w), int(person[p2][0]*h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kp in person:
            if kp[2] > 0.2:
                x, y = int(kp[1]*w), int(kp[0]*h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return frame

# Angle calculation
def calculate_angle(a, b, c):
    if a[2] < 0.2 or b[2] < 0.2 or c[2] < 0.2:
        return None
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

# Posture analysis
def analyze_posture(person):
    angles = {
        "left_elbow": calculate_angle(person[5], person[7], person[9]),
        "right_elbow": calculate_angle(person[6], person[8], person[10]),
        "left_shoulder": calculate_angle(person[11], person[5], person[7]),
        "right_shoulder": calculate_angle(person[12], person[6], person[8]),
        "left_knee": calculate_angle(person[11], person[13], person[15]),
        "right_knee": calculate_angle(person[12], person[14], person[16]),
        "left_wrist": person[9][2],
        "right_wrist": person[10][2]
    }
    corrections = {
        "elbow_drop": (
            ("Left " if angles["left_elbow"] and angles["left_elbow"] < 40 else "") +
            ("Right" if angles["right_elbow"] and angles["right_elbow"] < 40 else "")
        ).strip(),
        "bad_stance": "Yes" if (angles["left_knee"] and angles["left_knee"] > 170) or 
                             (angles["right_knee"] and angles["right_knee"] > 170) else "No"
    }
    return angles, corrections

# Glove detection
def detect_gloves(person):
    gloves = []
    if person[9][2] > 0.3: gloves.append(("Left Glove", person[9]))
    if person[10][2] > 0.3: gloves.append(("Right Glove", person[10]))
    return gloves

# Punch type
def detect_punch_type(person):
    lw, rw, le, re = person[9], person[10], person[7], person[8]
    punch = []
    if rw[2] > 0.3 and rw[1] > re[1]: punch.append("Right Jab")
    elif rw[2] > 0.3 and rw[0] < re[0]: punch.append("Right Hook")
    if lw[2] > 0.3 and lw[1] < le[1]: punch.append("Left Jab")
    elif lw[2] > 0.3 and lw[0] < le[0]: punch.append("Left Hook")
    return ", ".join(punch) if punch else "Guard"

# Annotate gloves
def annotate(frame, gloves):
    h, w, _ = frame.shape
    for name, (y, x, c) in gloves:
        cx, cy = int(x * w), int(y * h)
        cv2.putText(frame, name, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
    return frame

# Process video
def process_video(video_path, model, resize_w=480, skip_rate=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize_h = int(resize_w * orig_h / orig_w)

    output_path = tempfile.mktemp(suffix='.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), max(5, fps // skip_rate), (resize_w, resize_h))

    logs, frame_idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % skip_rate != 0:
            frame_idx += 1
            continue

        frame = cv2.resize(frame, (resize_w, resize_h))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_keypoints = detect_poses(rgb, model)
        persons = filter_top_two_persons(raw_keypoints)

        for i, person in enumerate(persons):
            gloves = detect_gloves(person)
            punch = detect_punch_type(person)
            posture, corrections = analyze_posture(person)

            logs.append({
                "frame": frame_idx,
                "person": i,
                "punch": punch,
                **posture,
                **corrections
            })

            frame = annotate(frame, gloves)
        frame = draw_skeleton(frame, persons)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path, pd.DataFrame(logs)

# Streamlit UI
st.set_page_config(layout="centered", page_title="🥊 Boxing Pose Analyzer")
st.title("🥊 Boxing Analyzer with Pose + Punch Detection")

model = load_model()
video_file = st.file_uploader("📤 Upload a Boxing Video", type=["mp4", "mov", "avi"])

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp.flush()
        st.video(tmp.name)
        with st.spinner("⏳ Processing video..."):
            annotated_path, df = process_video(tmp.name, model)

    st.success("✅ Done! Here's the output video:")
    st.video(annotated_path)

    with open(annotated_path, "rb") as f:
        st.download_button("⬇️ Download Annotated Video", f, file_name="boxing_annotated.mp4", mime="video/mp4")

    st.subheader("📊 Punch & Posture Data")
    st.dataframe(df.head())

    csv_io = io.StringIO()
    df.to_csv(csv_io, index=False)
    st.download_button("📥 Download CSV", csv_io.getvalue(), "boxing_analysis.csv", mime="text/csv")

    if st.button("🧠 Train SVM on Punch Labels"):
        if 'punch' in df.columns:
            X = df[['frame', 'person']]
            y = df['punch']
            clf = svm.SVC()
            clf.fit(X, y)
            model_path = "/tmp/svm_punch_model.joblib"
            dump(clf, model_path)
            st.success(f"✅ Trained and saved to {model_path}")
