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

st.set_option('client.showErrorDetails', True)
st.title("🥊 Boxing Analyzer with Punches, Posture & Gloves")

@st.cache_resource
def load_model():
    os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'
    return hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

model = load_model()

def extract_keypoints(results):
    people = []
    raw = results['output_0'].numpy()[0]  # shape (6, 56)
    for person_data in raw:
        keypoints = np.array(person_data[:51]).reshape(17, 3)  # y, x, confidence
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

def detect_gloves(keypoints):
    gloves = []
    for kp in keypoints:
        lw = kp[9]   # left wrist
        rw = kp[10]  # right wrist

        left_glove = "yes" if lw[2] > 0.3 else "no"
        right_glove = "yes" if rw[2] > 0.3 else "no"

        gloves.append((left_glove, right_glove))
    return gloves

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def draw_annotations(frame, keypoints, punches, postures, gloves):
    h, w = frame.shape[:2]
    annotated_frame = frame.copy()

    for i, kp in enumerate(keypoints):
        # Draw keypoints
        for (y, x, s) in kp:
            if s > 0.2:
                cx, cy = int(x * w), int(y * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 0), -1)

        # Draw skeleton
        for (p1, p2) in SKELETON_EDGES:
            y1, x1, s1 = kp[p1]
            y2, x2, s2 = kp[p2]
            if s1 > 0.2 and s2 > 0.2:
                pt1 = int(x1 * w), int(y1 * h)
                pt2 = int(x2 * w), int(y2 * h)
                cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 2)

        # Draw glove boxes
        for side, wrist_idx in zip(["L", "R"], [9, 10]):
            y, x, s = kp[wrist_idx]
            if s > 0.3:
                cx, cy = int(x * w), int(y * h)
                if 0 <= cx < w and 0 <= cy < h:
                    pad = 15
                    cv2.rectangle(annotated_frame, (cx - pad, cy - pad), (cx + pad, cy + pad), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"{side} Glove", (cx - pad, cy - pad - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Text annotations
        visible = [(y, x) for (y, x, s) in kp if s > 0.2]
        if visible:
            y_coords, x_coords = zip(*visible)
            min_x = int(min(x_coords) * w)
            max_y = int(max(y_coords) * h)
            label = f"{punches[i]}, {postures[i]}, L-Glove:{gloves[i][0]}, R-Glove:{gloves[i][1]}"
            cv2.putText(annotated_frame, label, (min_x, max_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated_frame

uploaded_files = st.file_uploader("Upload boxing MP4 videos", type=["mp4"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, uploaded_file.name)

        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(input_path)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        raw_output = os.path.join(temp_dir, "raw_output.mp4")
        out_writer = cv2.VideoWriter(raw_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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
            gloves = detect_gloves(keypoints, distance_thresh=0.1)
            annotated = draw_annotations(frame, keypoints, punches, postures, gloves)
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

        final_output = os.path.join(temp_dir, f"final_{uploaded_file.name}")
        ffmpeg.input(raw_output).output(final_output, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True)

        st.video(final_output)
        st.success("✅ Annotated video ready")

        with open(final_output, "rb") as f:
            st.download_button("📥 Download Annotated Video", f, file_name=f"annotated_{uploaded_file.name}", mime="video/mp4")

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
