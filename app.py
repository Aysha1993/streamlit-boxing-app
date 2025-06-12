# sort.py + MoveNet MultiPose Integration for Boxing Analyzer
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import streamlit as st
import tempfile
import os

# --- SORT Tracker ---
class Track:
    def __init__(self, bbox, track_id):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = np.array(bbox).reshape((4, 1))
        self.track_id = track_id

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].reshape((4,))

    def update(self, bbox):
        self.kf.update(np.array(bbox))

class Sort:
    def __init__(self, iou_threshold=0.3):
        self.trackers = []
        self.next_id = 0
        self.iou_threshold = iou_threshold

    def update(self, detections):
        updated_tracks = []
        for trk in self.trackers:
            trk.predict()

        iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
        for t, trk in enumerate(self.trackers):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = iou(det, trk.kf.x[:4].reshape((4,)))

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched, unmatched_dets, unmatched_trks = [], list(range(len(detections))), list(range(len(self.trackers)))
        for t, d in zip(*matched_indices):
            if iou_matrix[t, d] < self.iou_threshold: continue
            matched.append((t, d))
            unmatched_dets.remove(d)
            unmatched_trks.remove(t)

        for t, d in matched:
            self.trackers[t].update(detections[d])
            updated_tracks.append((self.trackers[t].track_id, detections[d]))

        for d in unmatched_dets:
            new_trk = Track(detections[d], self.next_id)
            self.next_id += 1
            self.trackers.append(new_trk)
            updated_tracks.append((new_trk.track_id, detections[d]))

        self.trackers = [trk for i, trk in enumerate(self.trackers) if i not in unmatched_trks]
        return updated_tracks

def iou(bb1, bb2):
    x1, y1, x2, y2 = bb1
    xx1, yy1, xx2, yy2 = bb2
    xi1, yi1, xi2, yi2 = max(x1, xx1), max(y1, yy1), min(x2, xx2), min(y2, yy2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (xx2 - xx1) * (yy2 - yy1)
    union_area = bb1_area + bb2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --- Movenet + Tracker ---
movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1").signatures['serving_default']
tracker = Sort()

def detect_persons(frame):
    input_img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
    outputs = movenet(input_img)
    kp_scores = outputs["output_0"].numpy()[:, 0, :, :]
    persons, boxes = [], []
    for kp in kp_scores:
        if kp[0, 2] < 0.1: continue
        kps = kp[:, :2]
        scores = kp[:, 2]
        min_x, min_y = np.min(kps[:, 0]), np.min(kps[:, 1])
        max_x, max_y = np.max(kps[:, 0]), np.max(kps[:, 1])
        bbox = [min_x, min_y, max_x, max_y]
        persons.append({'keypoints': kps, 'scores': scores, 'bbox': bbox})
        boxes.append(bbox)
    return persons, boxes

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        persons, boxes = detect_persons(frame)
        boxes_abs = [[int(b[0]*width), int(b[1]*height), int(b[2]*width), int(b[3]*height)] for b in boxes]
        tracks = tracker.update(boxes_abs)
        tracked = []
        for track_id, bbox in tracks:
            for person in persons:
                pb = person['bbox']
                abs_box = [int(pb[0]*width), int(pb[1]*height), int(pb[2]*width), int(pb[3]*height)]
                if np.allclose(abs_box, bbox, atol=20):
                    tracked.append((track_id, person))
                    break
        annotated = draw_ids(frame, tracked)
        out.write(annotated)
    cap.release()
    out.release()
    st.info(f"✅ SORT tracking video saved: {output_path}")

uploaded_files = st.file_uploader("Upload boxing video", type=["mp4", "avi", "mov"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Define a path for the uploaded video
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)

        # Save the uploaded video to disk
        with open(temp_video_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Now call your process_video function
        process_video(temp_video_path)



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


