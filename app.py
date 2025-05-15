import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
import ffmpeg
from sklearn import svm
from joblib import dump
import io

# Streamlit setup
st.set_option('client.showErrorDetails', True)
st.title("🥊 Boxing Analyzer with Punches, Posture & Gloves")

# Load MoveNet MultiPose model from TFHub
@st.cache_resource
def load_model():
    os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'
    return hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

model = load_model()

# Utility functions
def extract_keypoints(results):
    people = []
    raw = results['output_0'].numpy()[0]  # shape (6, 56)
    for person_data in raw:
        keypoints = np.array(person_data[:51]).reshape(17, 3)
        score = person_data[55]
        if score > 0.2 and np.mean(keypoints[:, 2]) > 0.2:
            people.append(keypoints.tolist())
    return people

def classify_punch(keypoints):
    result = []
    for kp in keypoints:
        lw, rw = kp[9], kp[10]
        ls, rs = kp[5], kp[6]
        le, re = kp[7], kp[8]

        if lw[2] > 0.2 and ls[2] > 0.2 and lw[0] < ls[0]:
            result.append("Left Jab")
        elif rw[2] > 0.2 and rs[2] > 0.2 and rw[0] < rs[0]:
            result.append("Right Jab")
        elif lw[2] > 0.2 and ls[2] > 0.2 and abs(lw[0] - ls[0]) > 0.1:
            result.append("Left Cross")
        elif rw[2] > 0.2 and rs[2] > 0.2 and abs(rw[0] - rs[0]) > 0.1:
            result.append("Right Cross")
        elif le[2] > 0.2 and abs(le[1] - lw[1]) > 0.1:
            result.append("Left Hook")
        elif re[2] > 0.2 and abs(re[1] - rw[1]) > 0.1:
            result.append("Right Hook")
        else:
            result.append("Guard")
    return result

def check_posture(keypoints):
    feedback = []
    for kp in keypoints:
        msgs = []
        if kp[7][0] > kp[11][0]: msgs.append("Left Elbow drop")
        if kp[8][0] > kp[12][0]: msgs.append("Right Elbow drop")
        if kp[5][0] > kp[11][0]: msgs.append("Left Shoulder drop")
        if kp[6][0] > kp[12][0]: msgs.append("Right Shoulder drop")
        if kp[15][0] < kp[13][0] - 0.05: msgs.append("Left Knee Bent")
        if kp[16][0] < kp[14][0] - 0.05: msgs.append("Right Knee Bent")
        if kp[9][0] > kp[7][0]: msgs.append("Left Wrist drop")
        if kp[10][0] > kp[8][0]: msgs.append("Right Wrist drop")
        if not msgs:
            msgs.append("Good Posture")
        feedback.append(", ".join(msgs))
    return feedback

def detect_gloves(keypoints, distance_thresh=0.1):
    gloves = []
    for kp in keypoints:
        lw, le = kp[9], kp[7]
        rw, re = kp[10], kp[8]

        def is_glove_present(wrist, elbow):
            if wrist[2] > 0.2 and elbow[2] > 0.2:
                dist = np.linalg.norm(np.array(wrist[:2]) - np.array(elbow[:2]))
                return dist > distance_thresh
            return False

        left_glove = "yes" if is_glove_present(lw, le) else "no"
        right_glove = "yes" if is_glove_present(rw, re) else "no"
        gloves.append(f"Gloves: L-{left_glove} R-{right_glove}")
    return gloves

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_annotations(frame, keypoints, punches, postures, gloves):
    h, w = frame.shape[:2]

    for i, kp in enumerate(keypoints):
        for (y, x, s) in kp:
            if s > 0.2:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        for (p1, p2) in SKELETON_EDGES:
            y1, x1, s1 = kp[p1]
            y2, x2, s2 = kp[p2]
            if s1 > 0.2 and s2 > 0.2:
                pt1 = int(x1 * w), int(y1 * h)
                pt2 = int(x2 * w), int(y2 * h)
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        for side, wrist_idx in zip(["L", "R"], [9, 10]):
            y, x, s = kp[wrist_idx]
            if s > 0.2:
                cx, cy = int(x * w), int(y * h)
                pad = 15
                cv2.rectangle(frame, (cx - pad, cy - pad), (cx + pad, cy + pad), (0, 0, 255), 2)
                cv2.putText(frame, f"{side} Glove", (cx - pad, cy - pad - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        visible_points = [(y, x) for (y, x, s) in kp if s > 0.2]
        if visible_points:
            y_coords, x_coords = zip(*visible_points)
            min_x = int(min(x_coords) * w)
            max_y = int(max(y_coords) * h)
            cv2.putText(frame, f"{punches[i]}, {postures[i]}", (min_x, max_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# Upload and process videos
uploaded_videos = st.file_uploader("Upload multiple boxing videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)

if uploaded_videos:
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

    all_punch_logs = []

    for video_file in uploaded_videos:
        video_name = video_file.name
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f"annotated_{video_name}"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_number = 0
        punch_logs = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints_with_scores = run_inference(model, image)
            annotated_image = draw_keypoints(image.copy(), keypoints_with_scores)
            punches = detect_punches(keypoints_with_scores)
            gloves = detect_gloves(keypoints_with_scores)
            posture = analyze_posture(keypoints_with_scores)

            frame_log = {
                "video": video_name,
                "frame": frame_number,
                "punches": punches,
                "gloves": gloves,
                "posture": posture
            }
            punch_logs.append(frame_log)

            for person_id, punch in punches.items():
                cv2.putText(annotated_image, f"Punch: {punch}", (10, 30 + person_id * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            for person_id, glove in gloves.items():
                cv2.putText(annotated_image, f"Glove: {glove}", (10, 80 + person_id * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for person_id, post in posture.items():
                cv2.putText(annotated_image, f"Posture: {post}", (10, 130 + person_id * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            out.write(annotated_image_bgr)

        cap.release()
        out.release()

        df = pd.DataFrame(punch_logs)
        all_punch_logs.extend(punch_logs)

        st.video(output_path)

    # Final combined CSV log
    full_log_df = pd.DataFrame(all_punch_logs)
    csv_path = "punch_log.csv"
    full_log_df.to_csv(csv_path, index=False)

    st.download_button("Download Punch Log CSV", data=open(csv_path, 'rb'), file_name="punch_log.csv", mime="text/csv")
