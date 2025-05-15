import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
from datetime import datetime
from moviepy.editor import ImageSequenceClip

# Load MoveNet Multipose
@st.cache_resource
def load_movenet_model():
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    return model.signatures['serving_default']

model = load_movenet_model()

# Define constants
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

def movenet_detect(image):
    img = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
    input_img = tf.cast(img, dtype=tf.int32)
    results = model(input_img)
    keypoints_with_scores = results['output_0'].numpy()
    return keypoints_with_scores[0]

def detect_punch_type(keypoints):
    results = []
    for person in keypoints:
        if person[0, 2] < 0.3:
            continue
        lw, rw = person[KEYPOINT_DICT['left_wrist']], person[KEYPOINT_DICT['right_wrist']]
        ls, rs = person[KEYPOINT_DICT['left_shoulder']], person[KEYPOINT_DICT['right_shoulder']]
        le, re = person[KEYPOINT_DICT['left_elbow']], person[KEYPOINT_DICT['right_elbow']]
        lh, rh = person[KEYPOINT_DICT['left_hip']], person[KEYPOINT_DICT['right_hip']]

        if lw[2] > 0.3 and lw[1] < ls[1] - 0.1:
            results.append("Left Jab")
        elif rw[2] > 0.3 and rw[1] > rs[1] + 0.1:
            results.append("Right Cross")
        elif lw[2] > 0.3 and le[2] > 0.3 and lw[1] > le[1]:
            results.append("Left Hook")
        elif rw[2] > 0.3 and re[2] > 0.3 and rw[1] < re[1]:
            results.append("Right Hook")
        elif lw[2] > 0.3 and rw[2] > 0.3 and lw[1] < ls[1] and rw[1] > rs[1]:
            results.append("Guard")
        elif lh[2] > 0.3 and rh[2] > 0.3 and lw[0] > lh[0] and rw[0] > rh[0]:
            results.append("Duck")
        else:
            results.append("Unknown")
    return results

def is_glove_present(wrist, elbow, thresh=0.08):
    if wrist[2] > 0.2 and elbow[2] > 0.2:
        dist = np.linalg.norm(np.array(wrist[:2]) - np.array(elbow[:2]))
        return dist > thresh
    return False

def check_posture(keypoints):
    posture_flags = []
    for person in keypoints:
        lw, rw = person[KEYPOINT_DICT['left_wrist']], person[KEYPOINT_DICT['right_wrist']]
        le, re = person[KEYPOINT_DICT['left_elbow']], person[KEYPOINT_DICT['right_elbow']]
        ls, rs = person[KEYPOINT_DICT['left_shoulder']], person[KEYPOINT_DICT['right_shoulder']]
        lh, rh = person[KEYPOINT_DICT['left_hip']], person[KEYPOINT_DICT['right_hip']]

        flags = []

        if le[2] > 0.2 and ls[2] > 0.2 and le[0] > ls[0] + 0.1:
            flags.append("Left Elbow Drop")
        if re[2] > 0.2 and rs[2] > 0.2 and re[0] > rs[0] + 0.1:
            flags.append("Right Elbow Drop")
        if lw[2] > 0.2 and rw[2] > 0.2 and lw[0] > lh[0] + 0.2 and rw[0] > rh[0] + 0.2:
            flags.append("Bad Stance")

        posture_flags.append(", ".join(flags) if flags else "Good")
    return posture_flags

def annotate_frame(frame, punches, gloves, posture):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    for i, label in enumerate(zip(punches, gloves, posture)):
        text = f"{label[0]} | Gloves: {label[1]} | Posture: {label[2]}"
        cv2.putText(frame, text, (30, 40 + i * 30), font, 0.6, color, 2)
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, logs = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_img, dtype=tf.uint8)
        keypoints = movenet_detect(input_tensor).reshape((6, 17, 3))

        punches = detect_punch_type(keypoints)
        posture = check_posture(keypoints)
        gloves = []

        for person in keypoints:
            lw, rw = person[KEYPOINT_DICT['left_wrist']], person[KEYPOINT_DICT['right_wrist']]
            le, re = person[KEYPOINT_DICT['left_elbow']], person[KEYPOINT_DICT['right_elbow']]
            gl = []
            if is_glove_present(lw, le): gl.append("Left")
            if is_glove_present(rw, re): gl.append("Right")
            gloves.append(",".join(gl) if gl else "None")

        annotated = annotate_frame(frame.copy(), punches, gloves, posture)
        frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        for punch, glove, post in zip(punches, gloves, posture):
            logs.append({
                "Punch": punch,
                "Glove": glove,
                "Posture": post
            })

    cap.release()
    return frames, pd.DataFrame(logs)

# Streamlit UI
st.title("🥊 Boxing Analyzer with MoveNet Multipose")

video_file = st.file_uploader("Upload boxing video", type=["mp4", "mov", "avi"])

if video_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())
    frames, df = process_video(temp_file.name)

    # Save annotated video
    output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
    clip = ImageSequenceClip(frames, fps=15)
    clip.write_videofile(output_path, codec="libx264", audio=False)

    st.video(output_path)
    st.dataframe(df)

    # Download punch log
    st.download_button("Download Punch Log CSV", df.to_csv(index=False), file_name="punch_log.csv")
