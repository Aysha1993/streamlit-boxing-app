import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import tempfile
import os
import base64

# --- Load MoveNet MultiPose Model ---
movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1").signatures['serving_default']

def detect_persons(frame):
    input_img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
    outputs = movenet(input_img)
    keypoints_with_scores = outputs['output_0'].numpy()  # shape: (1, 6, 56)

    persons, boxes = [], []
    for person in keypoints_with_scores[0]:
        if person[55] < 0.2:
            continue
        keypoints = person[:51].reshape((17, 3))
        bbox = person[51:55]
        kps = np.array([[kp[1], kp[0]] for kp in keypoints])
        scores = keypoints[:, 2]
        persons.append({'keypoints': kps, 'scores': scores, 'bbox': bbox})
        boxes.append(bbox)
    return persons, boxes

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
        x1, y1, x2, y2 = person['bbox']
        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for x, y in person['keypoints']:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
    return frame

def process_video(video_path, output_path="output_sort.mp4"):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    player_boxes = {}  # {player_id: bbox}
    initialized = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        persons, boxes = detect_persons(frame)
        boxes_abs = [[int(b[0]*width), int(b[1]*height), int(b[2]*width), int(b[3]*height)] for b in boxes]

        if not initialized and len(boxes_abs) >= 2:
            sorted_persons = sorted(zip(persons, boxes_abs), key=lambda x: (x[1][2]-x[1][0])*(x[1][3]-x[1][1]), reverse=True)
            player_boxes[1] = sorted_persons[0][1]
            player_boxes[2] = sorted_persons[1][1]
            initialized = True

        tracked = []
        for person in persons:
            pb = person['bbox']
            abs_box = [int(pb[0]*width), int(pb[1]*height), int(pb[2]*width), int(pb[3]*height)]

            matched_id = None
            for pid, ref_box in player_boxes.items():
                if iou(abs_box, ref_box) > 0.4:
                    matched_id = pid
                    player_boxes[pid] = abs_box
                    break

            if matched_id:
                tracked.append((matched_id, person))

        annotated = draw_ids(frame, tracked)
        out.write(annotated)

    cap.release()
    out.release()
    st.info(f"\u2705 Constant ID tracking video saved: {output_path}")

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
st.title("\ud83c\udfa5 Boxing Analyzer with Constant ID Tracking")

uploaded_files = st.file_uploader("Upload boxing video", type=["mp4", "avi", "mov"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_video_path, 'wb') as f:
            f.write(uploaded_file.read())

        output_path = os.path.join(temp_dir, "output_sort.mp4")
        process_video(temp_video_path, output_path)

        st.success("\u2705 Processed video with constant ID tracking.")
        play_video(output_path)

        with open(output_path, "rb") as file:
            st.download_button("\ud83d\udcc5 Download Tracked Video", file, "tracked_output.mp4", "video/mp4")

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
print("\u2705 requirements.txt saved")
