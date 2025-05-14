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
import io
import math

# Load MoveNet Multipose model
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    return model.signatures['serving_default']

# Detect poses from frame
def detect_poses(frame, model):
    input_size = 256
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = model(input_img)
    keypoints_with_scores = outputs['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
    return [person.tolist() for person in keypoints_with_scores if np.mean(person[:, 2]) > 0.2]

# Filter top 2 confident persons (assumed to be boxers)
def filter_top_two_persons(keypoints):
    scored = [(np.mean([s for (_, _, s) in kp]), idx) for idx, kp in enumerate(keypoints)]
    top_two = sorted(scored, reverse=True)[:2]
    return [keypoints[i] for (_, i) in top_two]

# Draw skeleton on frame
def draw_skeleton(frame, keypoints):
    height, width, _ = frame.shape
    keypoint_edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(9,10),
                      (11,12),(11,13),(13,15),(12,14),(14,16)]
    for person in keypoints:
        for edge in keypoint_edges:
            p1, p2 = person[edge[0]], person[edge[1]]
            if p1[2] > 0.2 and p2[2] > 0.2:
                x1, y1 = int(p1[1]*width), int(p1[0]*height)
                x2, y2 = int(p2[1]*width), int(p2[0]*height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kp in person:
            if kp[2] > 0.2:
                x, y = int(kp[1]*width), int(kp[0]*height)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return frame

# Utility: Calculate angle between 3 keypoints
def calculate_angle(a, b, c):
    if a[2] < 0.2 or b[2] < 0.2 or c[2] < 0.2:
        return None
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1, 1))
    return np.degrees(angle)

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
            "Left" if angles["left_elbow"] and angles["left_elbow"] < 40 else "" +
            " Right" if angles["right_elbow"] and angles["right_elbow"] < 40 else ""
        ).strip(),
        "bad_stance": "Yes" if (angles["left_knee"] and angles["left_knee"] > 170) or (angles["right_knee"] and angles["right_knee"] > 170) else "No"
    }
    return angles, corrections

# Detect gloves from wrists
def detect_gloves(person):
    gloves = []
    if person[9][2] > 0.3:
        gloves.append(('Left Glove', person[9]))
    if person[10][2] > 0.3:
        gloves.append(('Right Glove', person[10]))
    return gloves

# Simple punch detection
def detect_punch_type(person):
    left_wrist, right_wrist = person[9], person[10]
    left_elbow, right_elbow = person[7], person[8]
    punch = []

    if right_wrist[2] > 0.3 and right_wrist[1] > right_elbow[1]:
        punch.append("Right Jab")
    elif right_wrist[2] > 0.3 and right_wrist[0] < right_elbow[0]:
        punch.append("Right Hook")
    elif left_wrist[2] > 0.3 and left_wrist[1] < left_elbow[1]:
        punch.append("Left Jab")
    elif left_wrist[2] > 0.3 and left_wrist[0] < left_elbow[0]:
        punch.append("Left Hook")

    return ", ".join(punch) if punch else "Guard"

# Annotate gloves
def annotate(frame, gloves):
    height, width, _ = frame.shape
    for name, (y, x, c) in gloves:
        cx, cy = int(x * width), int(y * height)
        cv2.putText(frame, name, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
    return frame

# Process and annotate video
def process_video(input_path, model, resize_w=640, skip_rate=2):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps // skip_rate, (resize_w, int(resize_w * height / width)))

    frame_idx = 0
    log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_rate != 0:
            frame_idx += 1
            continue

        frame = cv2.resize(frame, (resize_w, int(resize_w * height / width)))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_keypoints = detect_poses(frame_rgb, model)
        keypoints = filter_top_two_persons(raw_keypoints)

        for i, person in enumerate(keypoints):
            gloves = detect_gloves(person)
            punch = detect_punch_type(person)
            posture, corrections = analyze_posture(person)

            log.append({
                "frame": frame_idx,
                "person": i,
                "punch": punch,
                **posture,
                **corrections
            })

            frame = annotate(frame, gloves)

        frame = draw_skeleton(frame, keypoints)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path, pd.DataFrame(log)

# Streamlit UI
st.title("🥊 Boxing Pose Estimator with Punch Detection, Posture & Corrections")

model = load_model()
video_file = st.file_uploader("Upload Boxing Video", type=["mp4", "mov"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    st.video(tfile.name)

    with st.spinner("Processing video..."):
        annotated_path, df = process_video(tfile.name, model)

    st.success("✅ Video processed!")
    st.video(annotated_path)

    with open(annotated_path, "rb") as f:
        st.download_button("⬇️ Download Annotated Video", f, file_name="annotated_output.mp4")

    st.dataframe(df.head())
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Analysis CSV", csv_buffer.getvalue(), file_name="boxing_analysis.csv", mime="text/csv")

    base_name = os.path.splitext(video_file.name)[0]
    model_dest = f"/tmp/{base_name}_svm_model.joblib"

    if st.button(f"Train SVM on {video_file.name}"):
        if 'punch' in df.columns:
            X = df[['frame', 'person']]
            y = df['punch']
            clf = svm.SVC()
            clf.fit(X, y)
            dump(clf, model_dest)
            st.success("✅ SVM trained and saved")
