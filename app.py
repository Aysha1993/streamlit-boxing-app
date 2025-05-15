import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
import shutil
import ffmpeg
from sklearn import svm
from joblib import dump
import io

# Load MoveNet Multipose model

@st.cache_resource
def load_model():
    model = hub.load("[https://tfhub.dev/google/movenet/multipose/lightning/1](https://tfhub.dev/google/movenet/multipose/lightning/1)")
    return model.signatures['serving\_default']

# Detect poses from frame

def detect_poses(frame, model):
    input_size = 256
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = model(input_img)
    keypoints_with_scores = outputs['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))


keypoints = []
for person in keypoints_with_scores:
    if np.mean(person[:, 2]) > 0.2:
        keypoints.append(person.tolist())
return keypoints


# Filter top 2 confident persons (assumed to be boxers)

def filter_top_two_persons(keypoints):
    scored = []
    for idx, kp in enumerate(keypoints):
        score = np.mean([s for (_, _, s) in kp])
        scored.append((score, idx))
        top_two = sorted(scored, reverse=True)[:2]
        
    return [keypoints[i] for (_, i) in top_two]
    

# Draw skeleton on frame

def draw_skeleton(frame, keypoints):
    
    height, width, _ = frame.shape
    keypoint_edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16)]
    for person in keypoints:   
        for edge in keypoint_edges:            
            p1 = person[edge[0]]
            p2 = person[edge[1]]
            if p1[2] > 0.2 and p2[2] > 0.2:
                x1, y1 = int(p1[1]*width), int(p1[0]*height)
                x2, y2 = int(p2[1]*width), int(p2[0]*height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for idx, kp in enumerate(person):
                if kp[2] > 0.2:
                    x, y = int(kp[1]*width), int(kp[0]*height)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return frame

# Detect gloves from wrist keypoints

def detect_gloves(keypoints):
    gloves = []
    for person in keypoints:
        left_wrist = person[9]
        right_wrist = person[10]
    if left_wrist[2] > 0.3:
        gloves.append(('Left Glove', left_wrist))
    if right_wrist[2] > 0.3:
        gloves.append(('Right Glove', right_wrist))
    return gloves

# Annotate detections

def annotate(frame, gloves):
    height, width, _ = frame.shape
    for name, (y, x, c) in gloves:
        cx, cy = int(x * width), int(y * height)
        cv2.putText(frame, name, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
    return frame

# Process and annotate video

def process_video(input_path, model):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_keypoints = detect_poses(frame_rgb, model)
        keypoints = filter_top_two_persons(raw_keypoints)  # Only top 2 confident persons assumed to be players
        gloves = detect_gloves(keypoints)
        frame = draw_skeleton(frame, keypoints)
        frame = annotate(frame, gloves)
        out.write(frame)

    cap.release()
    out.release()
    return out_path


# Streamlit UI

st.title("Boxing Pose Estimator with Glove Detection")
model = load_model()
video_file = st.file_uploader("Upload Boxing Video", type=["mp4", "mov"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    st.video(tfile.name)
    with st.spinner("Processing video..."):
        annotated_path = process_video(tfile.name, model)
        st.success("Video processed!")
        st.video(annotated_path)
    with open(annotated_path, "rb") as f:
        st.download_button("Download Annotated Video", f, file_name="annotated_output.mp4")


    df = pd.DataFrame(punch_log)
    st.dataframe(df)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download CSV", csv_buffer.getvalue(), file_name=f"{uploaded_file.name}_log.csv", mime="text/csv")

    base_name = os.path.splitext(uploaded_file.name)[0]
    model_dest = f"/tmp/{base_name}_svm_model.joblib"

    if st.button(f"Train SVM on {uploaded_file.name}"):
        if 'punch' in df.columns:
            X = df[['frame', 'person']]
            y = df['punch']
            clf = svm.SVC()
            clf.fit(X, y)
            dump(clf, model_dest)
            st.success("SVM trained and saved ✅")
