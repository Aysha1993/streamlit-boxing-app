import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
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
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

class Sort:
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_assignment(-iou_matrix)
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def linear_assignment(cost_matrix):
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))




import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from sort import Sort
import csv
import os

movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

def run_movenet(image):
    img = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = movenet.signatures['serving_default'](input_img)
    keypoints = outputs['output_0'].numpy()
    return keypoints[0]

def keypoints_to_bbox(keypoints):
    valid = keypoints[:, 2] > 0.2
    x = keypoints[valid, 0]
    y = keypoints[valid, 1]
    if len(x) == 0:
        return None
    return [np.min(x), np.min(y), np.max(x), np.max(y)]

def classify_punch(keypoints):
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]

    if left_wrist[1] < left_elbow[1]:
        return "Left Jab"
    elif right_wrist[1] < right_elbow[1]:
        return "Right Cross"
    else:
        return "Punch"

def check_posture(keypoints):
    messages = []
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_knee = keypoints[13]
    right_knee = keypoints[14]

    if left_elbow[1] > keypoints[5][1] + 30:
        messages.append("Left elbow dropped")
    if right_elbow[1] > keypoints[6][1] + 30:
        messages.append("Right elbow dropped")
    if abs(left_knee[1] - right_knee[1]) < 10:
        messages.append("Flat stance")

    return ", ".join(messages)

def annotate_frame(frame, keypoints, track_id, punch_type, posture_status):
    for x, y, conf in keypoints:
        if conf > 0.2:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    if punch_type:
        cv2.putText(frame, f"{punch_type}", (int(keypoints[0][0]), int(keypoints[0][1]) - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if posture_status:
        cv2.putText(frame, f"{posture_status}", (int(keypoints[0][0]), int(keypoints[0][1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
    cv2.putText(frame, f"Boxer {track_id}", (int(keypoints[0][0]), int(keypoints[0][1]) - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def extract_pose_and_track(video_path, output_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = Sort()
    punch_log = []
    frame_id = 0
    id_to_pose = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints_data = run_movenet(rgb_frame)

        dets, poses = [], []
        for i in range(keypoints_data.shape[0]):
            keypoints = keypoints_data[i]
            bbox = keypoints_to_bbox(keypoints)
            if bbox:
                dets.append([*bbox, 1.0])
                poses.append(keypoints)

        if len(dets) > 0:
            tracks = tracker.update(np.array(dets))
            for i, track in enumerate(tracks):
                x1, y1, x2, y2, track_id = track
                if int(track_id) not in id_to_pose:
                    if len(id_to_pose) < 2:
                        id_to_pose[int(track_id)] = True
                if int(track_id) in id_to_pose:
                    keypoints = poses[i]
                    punch_type = classify_punch(keypoints)
                    posture_status = check_posture(keypoints)
                    annotate_frame(frame, keypoints, int(track_id), punch_type, posture_status)
                    punch_log.append([frame_id, int(track_id), punch_type, posture_status])

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'boxer_id', 'punch_type', 'posture_issues'])
        writer.writerows(punch_log)


import streamlit as st
import os
from boxing_utils import extract_pose_and_track

st.title("🥊 Boxing Analyzer - Referee Filter + Punch & Posture Detection")

video_file = st.file_uploader("Upload a boxing video (.mp4)", type=['mp4'])

if video_file:
    with open("input_video.mp4", "wb") as f:
        f.write(video_file.read())
    st.video("input_video.mp4")

    if st.button("Run Analyzer"):
        os.makedirs("outputs", exist_ok=True)
        extract_pose_and_track("input_video.mp4", "outputs/annotated_video.mp4", "outputs/punch_log.csv")
        st.success("✅ Processing complete!")

        st.video("outputs/annotated_video.mp4")

        with open("outputs/punch_log.csv", "r") as f:
            st.download_button("Download Punch Log CSV", f, "punch_log.csv", "text/csv")



requirements = '''streamlit
opencv-python-headless
tensorflow
tensorflow-hub
ffmpeg-python
pyngrok
filterpy
numpy
'''
