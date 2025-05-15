import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import tempfile
import os
import pandas as pd
import ffmpeg

st.set_page_config(layout="wide")
st.title("🥊 Boxing Analyzer with Pose, Punch, Glove & Posture Detection")

# Load MoveNet MultiPose
@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    return model.signatures['serving_default']

movenet = load_model()

# Keypoint drawing configuration
EDGES = [
    (0, 1), (1, 3), (0, 2), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def detect_pose(image):
    input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet(input_image)
    return outputs['output_0'].numpy()[0]

def draw_skeleton(frame, keypoints, punches, posture_issues):
    h, w, _ = frame.shape
    for i, kp in enumerate(keypoints):
        y, x, c = kp
        if c > 0.3:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    for p1, p2 in EDGES:
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > 0.3 and c2 > 0.3:
            pt1 = int(x1 * w), int(y1 * h)
            pt2 = int(x2 * w), int(y2 * h)
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    y0 = 20
    for txt in punches + posture_issues:
        cv2.putText(frame, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y0 += 20
    return frame

def classify_punch(keypoints):
    punches = []
    l_sh, r_sh = keypoints[5], keypoints[6]
    l_el, r_el = keypoints[7], keypoints[8]
    l_wr, r_wr = keypoints[9], keypoints[10]

    def is_jab(sh, el, wr): return abs(wr[1] - sh[1]) < 0.1 and wr[0] < el[0]
    def is_cross(sh, el, wr): return abs(wr[1] - sh[1]) < 0.1 and wr[0] > el[0]
    def is_hook(sh, el, wr): return abs(wr[0] - el[0]) < 0.1 and abs(wr[1] - el[1]) > 0.05

    if is_jab(l_sh, l_el, l_wr): punches.append("Left Jab")
    if is_cross(r_sh, r_el, r_wr): punches.append("Right Cross")
    if is_hook(l_sh, l_el, l_wr): punches.append("Left Hook")
    if is_hook(r_sh, r_el, r_wr): punches.append("Right Hook")
    return punches

def posture_analysis(kp):
    issues = []
    if kp[7][1] > kp[5][1] + 0.1: issues.append("Left Elbow Drop")
    if kp[8][1] > kp[6][1] + 0.1: issues.append("Right Elbow Drop")
    if abs(kp[13][0] - kp[14][0]) < 0.1: issues.append("Bad Stance")
    return issues

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames

def save_video_ffmpeg(frames, output_path, fps=30):
    temp_path = "temp_output.avi"
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for f in frames: out.write(f)
    out.release()
    ffmpeg.input(temp_path).output(output_path, vcodec='libx264', pix_fmt='yuv420p').run(overwrite_output=True)
    os.remove(temp_path)

def save_csv(logs, path):
    df = pd.DataFrame(logs)
    df.to_csv(path, index=False)

# Upload and Process Video
video_file = st.file_uploader("Upload a boxing video", type=["mp4", "mov", "avi"])
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    st.info("⏳ Extracting frames...")
    frames = extract_frames(tfile.name)
    logs, annotated_frames = [], []

    for idx, frame in enumerate(frames):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detect_pose(img_rgb)

        for person in detections:
            kp = np.reshape(person, (17, 3))
            if np.mean(kp[:, 2]) < 0.2: continue
            punches = classify_punch(kp)
            issues = posture_analysis(kp)

            annotated = draw_skeleton(frame.copy(), kp, punches, issues)
            annotated_frames.append(annotated)

            logs.append({
                "frame": idx,
                "punches": ";".join(punches),
                "posture_issues": ";".join(issues)
            })
            break  # Only one person per frame (player focus)

    out_path = "output_annotated.mp4"
    csv_path = "punch_log.csv"
    st.info("💾 Saving annotated video...")
    save_video_ffmpeg(annotated_frames, out_path)
    save_csv(logs, csv_path)

    st.success("✅ Done! Preview below:")
    st.video(out_path)

    with open(out_path, "rb") as f:
        st.download_button("📥 Download Annotated Video", f, "boxing_annotated.mp4")

    with open(csv_path, "rb") as f:
        st.download_button("📥 Download Punch & Posture Log CSV", f, "boxing_log.csv")
