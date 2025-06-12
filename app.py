import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import tempfile
import os
import base64
from collections import defaultdict

# --- Load MoveNet MultiPose Model ---
movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1").signatures['serving_default']

def detect_persons(frame):
    input_img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
    outputs = movenet(input_img)
    keypoints_with_scores = outputs['output_0'].numpy()  # shape: (1, 6, 56)

    persons = []
    for person in keypoints_with_scores[0]:
        if person[55] < 0.2:
            continue
        keypoints = person[:51].reshape((17, 3))
        bbox = person[51:55]  # [ymin, xmin, ymax, xmax]
        kps = np.array([[kp[1], kp[0]] for kp in keypoints])  # (x, y)
        scores = keypoints[:, 2]
        persons.append({
            'keypoints': kps,
            'scores': scores,
            'bbox_norm': [bbox[1], bbox[0], bbox[3], bbox[2]]  # [xmin, ymin, xmax, ymax]
        })
    return persons

def iou(bb1, bb2):
    x1, y1, x2, y2 = bb1
    xx1, yy1, xx2, yy2 = bb2
    xi1, yi1 = max(x1, xx1), max(y1, yy1)
    xi2, yi2 = min(x2, xx2), min(y2, yy2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (xx2 - xx1) * (yy2 - yy1)
    union_area = bb1_area + bb2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def draw_ids(frame, tracked):
    h, w, _ = frame.shape
    for tid, person in tracked:
        x1n, y1n, x2n, y2n = person['bbox_norm']
        x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for x, y in person['keypoints']:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
    return frame

def estimate_punch_activity(keypoints_seq):
    count = 0
    for i in range(1, len(keypoints_seq)):
        prev = keypoints_seq[i-1]
        curr = keypoints_seq[i]
        if prev is None or curr is None:
            continue
        lw_dist = np.linalg.norm(curr[9] - prev[9])
        rw_dist = np.linalg.norm(curr[10] - prev[10])
        if lw_dist > 0.03 or rw_dist > 0.03:
            count += 1
    return count

def process_video(video_path, output_path="output_sort.mp4"):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    player_boxes = {}
    initialized = False
    person_history = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        persons = detect_persons(frame)
        person_with_abs_boxes = []
        for person in persons:
            x1n, y1n, x2n, y2n = person['bbox_norm']
            x1, y1, x2, y2 = int(x1n * width), int(y1n * height), int(x2n * width), int(y2n * height)
            person_with_abs_boxes.append((person, [x1, y1, x2, y2]))

        tracked = []
        for person, abs_box in person_with_abs_boxes:
            matched_id = None
            for pid, ref_box in player_boxes.items():
                if iou(abs_box, ref_box) > 0.4:
                    matched_id = pid
                    player_boxes[pid] = abs_box
                    break

            if not matched_id:
                new_pid = max(player_boxes.keys(), default=0) + 1
                player_boxes[new_pid] = abs_box
                matched_id = new_pid

            tracked.append((matched_id, person))
            person_history[matched_id].append(person['keypoints'])

        annotated = draw_ids(frame, tracked)
        out.write(annotated)

    cap.release()
    out.release()

    punch_counts = {pid: estimate_punch_activity(seq) for pid, seq in person_history.items()}
    top_2_punchers = sorted(punch_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    valid_pids = set(pid for pid, _ in top_2_punchers)

    # Rerun to create new video with only top 2 punchers (boxers)
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    player_boxes.clear()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        persons = detect_persons(frame)
        person_with_abs_boxes = []
        for person in persons:
            x1n, y1n, x2n, y2n = person['bbox_norm']
            x1, y1, x2, y2 = int(x1n * width), int(y1n * height), int(x2n * width), int(y2n * height)
            person_with_abs_boxes.append((person, [x1, y1, x2, y2]))

        tracked = []
        for person, abs_box in person_with_abs_boxes:
            for pid, seq in person_history.items():
                if person in seq and pid in valid_pids:
                    tracked.append((pid, person))
                    break

        annotated = draw_ids(frame, tracked)
        out.write(annotated)

    cap.release()
    out.release()
    st.info(f"✅ Tracked video with punch-based PID filtering saved: {output_path}")

def play_video(video_path):
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        b64_encoded = base64.b64encode(video_bytes).decode()
        video_html = f'''
            <video width="700" controls>
                <source src="data:video/mp4;base64,{b64_encoded}" type="video/mp4">
            </video>
        '''
        st.markdown(video_html, unsafe_allow_html=True)

# --- Streamlit UI ---
st.title("🎥 Boxing Analyzer with Punch-Based Boxer Detection")

uploaded_files = st.file_uploader("Upload boxing video", type=["mp4", "avi", "mov"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_video_path, 'wb') as f:
            f.write(uploaded_file.read())

        output_path = os.path.join(temp_dir, "output_sort.mp4")
        process_video(temp_video_path, output_path)

        st.success("✅ Processed video with punch-based PID filtering.")
        play_video(output_path)

        with open(output_path, "rb") as file:
            st.download_button("📥 Download Tracked Video", file, "tracked_output.mp4", "video/mp4")

# --- requirements.txt generator ---
requirements = '''streamlit
tensorflow
tensorflow_hub
opencv-python-headless
filterpy
pandas
numpy
scikit-learn
joblib
ffmpeg-python
tqdm
seaborn
lightgbm
imbalanced-learn
plotly
matplotlib
'''

with open("requirements.txt", "w") as f:
    f.write(requirements)
print("✅ requirements.txt saved")

