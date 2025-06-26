import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import ffmpeg
import io
import json
import tensorflow as tf
import tensorflow_hub as hub
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib
# from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np
import time


# Streamlit setup
st.set_option('client.showErrorDetails', True)
st.title("🥊 Boxing Analyzer App")


# Load MoveNet MultiPose model from TFHub
@st.cache_resource
def load_model():
    os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'
    return hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

model = load_model()

# Utility functions
def extract_keypoints(results):
    people = []
    raw = results['output_0'].numpy()  # shape (1, 6, 56)
    for person_data in raw[0]:
        keypoints = np.array(person_data[:51]).reshape(17, 3)
        score = person_data[55]
        if score > 0.2 and np.mean(keypoints[:, 2]) > 0.2:
            people.append(keypoints.tolist())
    return people

import numpy as np

# Global state: person_id -> punch state tracker
person_states = {}

# Tunable thresholds
VELOCITY_THRESHOLD = 0.1  # adjust based on pixel movement per frame
HOOK_ANGLE_THRESHOLD = 60  # elbow angle in degrees for hook detection

def calculate_velocity(prev_point, curr_point):
    return np.linalg.norm(np.array(curr_point) - np.array(prev_point))

def calculate_elbow_angle(shoulder, elbow, wrist):
    a = np.array(shoulder)
    b = np.array(elbow)
    c = np.array(wrist)
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

confidence_threshold=0.3
#pose based filtering
def is_punching_pose(person):
    """
    Heuristically detect punch-like posture: one arm extended forward
    """
    def arm_extended(shoulder_idx, elbow_idx, wrist_idx):
        s = person[shoulder_idx]
        e = person[elbow_idx]
        w = person[wrist_idx]

        # Confidence check
        if s[2] < confidence_threshold or e[2] < confidence_threshold or w[2] < confidence_threshold:
            return False

        # Vector direction
        sx, sy = s[0], s[1]
        ex, ey = e[0], e[1]
        wx, wy = w[0], w[1]

        # Approximate straightness of arm using angle between segments
        vec1 = np.array([ex - sx, ey - sy])
        vec2 = np.array([wx - ex, wy - ey])
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return False

        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

        # Angle near 180 means straight arm (punch-like)
        return angle > 150

    # Return True if at least one arm is extended
    return arm_extended(5, 7, 9) or arm_extended(6, 8, 10)  # Left or Right arm


# Map from joint name to index in MoveNet
keypoint_index = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# --- Globals for punch tracking ---
last_punch_time = {}     # {person_id: timestamp}
last_punch_label = {}    # {person_id: last punch type}
PUNCH_COOLDOWN = 0.5     # seconds

# --- Angle Helper ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# --- Cooldown logic ---
def allow_punch(person_id, timestamp):
    last_time = last_punch_time.get(person_id, -999)
    if timestamp - last_time > PUNCH_COOLDOWN:
        last_punch_time[person_id] = timestamp
        return True
    return False

# --- Punch Detection ---
def detect_punch(person_id, keypoints, timestamp):
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 5, 6
    LEFT_ELBOW, RIGHT_ELBOW = 7, 8
    LEFT_WRIST, RIGHT_WRIST = 9, 10
    LEFT_HIP, RIGHT_HIP = 11, 12

    nose = keypoints[NOSE][:2]
    lw, rw = keypoints[LEFT_WRIST][:2], keypoints[RIGHT_WRIST][:2]
    le, re = keypoints[LEFT_ELBOW][:2], keypoints[RIGHT_ELBOW][:2]
    ls, rs = keypoints[LEFT_SHOULDER][:2], keypoints[RIGHT_SHOULDER][:2]
    lh, rh = keypoints[LEFT_HIP][:2], keypoints[RIGHT_HIP][:2]

    dist_lw_nose = np.linalg.norm(lw - nose)
    dist_rw_nose = np.linalg.norm(rw - nose)
    left_elbow_angle = calculate_angle(ls, le, lw)
    right_elbow_angle = calculate_angle(rs, re, rw)
    left_shoulder_angle = calculate_angle(le, ls, lh)
    right_shoulder_angle = calculate_angle(re, rs, rh)

    head_height = nose[1]
    punch_type = "None"

    # Rule-based detection
    if dist_lw_nose > 70 and left_elbow_angle > 140 and np.linalg.norm(lw - le) > 30:
        punch_type = "Left Jab"
    elif dist_rw_nose > 70 and right_elbow_angle > 140 and np.linalg.norm(rw - re) > 30:
        punch_type = "Right Cross"
    elif ((left_elbow_angle < 100 and left_shoulder_angle > 80) or
          (right_elbow_angle < 100 and right_shoulder_angle > 80)):
        punch_type = "Hook"
    elif head_height > rs[1] + 40 and head_height > ls[1] + 40:
        punch_type = "Duck"
    elif dist_lw_nose < 50 and dist_rw_nose < 50:
        punch_type = "Guard"

    is_new_punch = False
    # Cooldown check
    if punch_type != "None" and allow_punch(person_id, timestamp):
        last_punch_label[person_id] = punch_type
        is_new_punch = True

    return last_punch_label.get(person_id, "None"), is_new_punch



# # Global punch cooldown tracker
# last_punch_time = {}  # {person_id: timestamp}
# PUNCH_COOLDOWN = 0.5  # seconds, increased to prevent overcounting

# def allow_punch(person_id, timestamp):
#     """Allow punch only if enough time passed since last."""
#     last_time = last_punch_time.get(person_id, -999)
#     if timestamp - last_time > PUNCH_COOLDOWN:
#         last_punch_time[person_id] = timestamp
#         return True
#     return False

# def calculate_angle(a, b, c):
#     """Calculate angle (in degrees) between points a-b-c."""
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
#     return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# def detect_punch(person_id, keypoints, timestamp):
#     """Detect punch type using keypoints and cooldown control."""
#     NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 5, 6
#     LEFT_ELBOW, RIGHT_ELBOW = 7, 8
#     LEFT_WRIST, RIGHT_WRIST = 9, 10
#     LEFT_HIP, RIGHT_HIP = 11, 12

#     # Get coordinates
#     nose = keypoints[NOSE][:2]
#     lw, rw = keypoints[LEFT_WRIST][:2], keypoints[RIGHT_WRIST][:2]
#     le, re = keypoints[LEFT_ELBOW][:2], keypoints[RIGHT_ELBOW][:2]
#     ls, rs = keypoints[LEFT_SHOULDER][:2], keypoints[RIGHT_SHOULDER][:2]
#     lh, rh = keypoints[LEFT_HIP][:2], keypoints[RIGHT_HIP][:2]

#     # Distances and angles
#     dist_lw_nose = np.linalg.norm(lw - nose)
#     dist_rw_nose = np.linalg.norm(rw - nose)
#     left_elbow_angle = calculate_angle(ls, le, lw)
#     right_elbow_angle = calculate_angle(rs, re, rw)
#     left_shoulder_angle = calculate_angle(le, ls, lh)
#     right_shoulder_angle = calculate_angle(re, rs, rh)

#     head_height = nose[1]
#     punch_type = "None"

#     # Rule-based detection (tightened thresholds)
#     if dist_lw_nose > 70 and left_elbow_angle > 140 and np.linalg.norm(lw - le) > 30:
#         punch_type = "Left Jab"
#     elif dist_rw_nose > 70 and right_elbow_angle > 140 and np.linalg.norm(rw - re) > 30:
#         punch_type = "Right Cross"
#     elif ((left_elbow_angle < 100 and left_shoulder_angle > 80) or
#           (right_elbow_angle < 100 and right_shoulder_angle > 80)):
#         punch_type = "Hook"
#     elif head_height > rs[1] + 40 and head_height > ls[1] + 40:
#         punch_type = "Duck"
#     elif dist_lw_nose < 50 and dist_rw_nose < 50:
#         punch_type = "Guard"

#     # Final filter based on cooldown
#     if punch_type != "None" and allow_punch(person_id, timestamp):
#         return punch_type
#     return "None"

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

def detect_gloves_by_color_and_shape(frame, keypoints, confidence_threshold=0.3, crop_size=30):
    """
    Detect boxing gloves using wrist keypoints and color/shape, only when punch posture is detected.

    Args:
        frame: BGR image (numpy array)
        keypoints: List of people, each a list of 17 keypoints (x, y, confidence)
        confidence_threshold: Minimum confidence for keypoints
        crop_size: Half-size of square patch to crop around wrist

    Returns:
        List of glove detections: [{'left_glove': True/False, 'right_glove': True/False}, ...]
    """
    glove_detections = []
    #pose based filtering
    def is_punching_pose(person):
        """
        Heuristically detect punch-like posture: one arm extended forward
        """
        def arm_extended(shoulder_idx, elbow_idx, wrist_idx):
            s = person[shoulder_idx]
            e = person[elbow_idx]
            w = person[wrist_idx]

            # Confidence check
            if s[2] < confidence_threshold or e[2] < confidence_threshold or w[2] < confidence_threshold:
                return False

            # Vector direction
            sx, sy = s[0], s[1]
            ex, ey = e[0], e[1]
            wx, wy = w[0], w[1]

            # Approximate straightness of arm using angle between segments
            vec1 = np.array([ex - sx, ey - sy])
            vec2 = np.array([wx - ex, wy - ey])
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return False

            cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

            # Angle near 180 means straight arm (punch-like)
            return angle > 150

        # Return True if at least one arm is extended
        return arm_extended(5, 7, 9) or arm_extended(6, 8, 10)  # Left or Right arm

    for person in keypoints:
        h, w, _ = frame.shape

        # if not is_punching_pose(person):
        #     glove_detections.append({'left_glove': False, 'right_glove': False})
        #     continue  # Skip glove check if not punching

        def crop_wrist_region(wrist_index):
            kp = person[wrist_index]
            if kp[2] < confidence_threshold:
                return None
            x = int(kp[0] * w)
            y = int(kp[1] * h)
            x1, y1 = max(0, x - crop_size), max(0, y - crop_size)
            x2, y2 = min(w, x + crop_size), min(h, y + crop_size)
            return frame[y1:y2, x1:x2]

        def is_glove(region):
            if region is None or region.size == 0:
                return False

            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            glove_colors = {
                'red': ((0, 70, 50), (10, 255, 255)),
                'blue': ((100, 100, 50), (140, 255, 255)),
                'black': ((0, 0, 0), (180, 255, 60)),
            }

            mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in glove_colors.values():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask_total = cv2.bitwise_or(mask_total, mask)

            glove_ratio = np.sum(mask_total > 0) / mask_total.size
            if glove_ratio < 0.2:
                return False

            contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    return True
            return False

        left_crop = crop_wrist_region(9)
        right_crop = crop_wrist_region(10)

        left_glove = is_glove(left_crop)
        right_glove = is_glove(right_crop)

        glove_detections.append({
            'left_glove': left_glove,
            'right_glove': right_glove
        })

    return glove_detections


def is_likely_coach(keypoints,
                    min_avg_conf=0.5,
                    min_bbox_height_ratio=0.3,
                    min_keypoints_detected=8):
    """
    Determine if a detected person is likely a coach or irrelevant detection.
    """
    confidences = [kp[2] for kp in keypoints]
    # st.info(f"{confidences}")
    avg_conf = np.mean(confidences)
    num_valid_kps = sum(c > 0.2 for c in confidences)
    ys = [kp[0] for kp in keypoints if kp[2] > 0.2]
    bbox_height = max(ys) - min(ys) if ys else 0

    is_coach = (
        avg_conf < min_avg_conf or
        num_valid_kps < min_keypoints_detected or
        bbox_height < min_bbox_height_ratio
    )
    # Debug info
    st.info(f"[DEBUG] CoachCheck → avg_conf={avg_conf:.2f} valid_kps={num_valid_kps}, bbox_height={bbox_height:.2f} → is_coach={is_coach}")

    return is_coach


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_EDGES = [
    (0, 1), (1, 3), (0, 2), (2, 4),         # Face
    (5, 7), (7, 9), (6, 8), (8, 10),        # Arms
    (5, 6), (5, 11), (6, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16), # Legs
    (11, 12)                                # Hip line
]


def draw_annotations(frame, keypoints, punches, postures, glove_detections, h, w,person_id):
    y_offset = 30
    line_height = 20

    valid_detections = []
    for idx, (kp_raw, punch, posture, glovedetected,pid) in enumerate(zip(keypoints, punches, postures, glove_detections,person_id)):
        person = kp_raw  # use the current person only
        # if not is_punching_pose(person):
        #     #st.info(f"Skipping Person {idx+1} - Not Punching")
        #     continue
        if pid in [2, 3]:  # skip referees
            continue

        kp = np.array(kp_raw).reshape(-1, 3).tolist()

        #Draw keypoints
        for i, (y, x, s) in enumerate(kp):
            if s > 0.2:
                cx, cy = int(x * w), int(y * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(frame, KEYPOINT_NAMES[i], (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        #Draw skeleton
        for (p1, p2) in SKELETON_EDGES:
            y1, x1, s1 = kp[p1]
            y2, x2, s2 = kp[p2]
            if s1 > 0.2 and s2 > 0.2:
                pt1 = int(x1 * w), int(y1 * h)
                pt2 = int(x2 * w), int(y2 * h)
                if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        #Draw gloves
        for side, kp_idx in [('left', 9), ('right', 10)]:
            if glovedetected.get(f"{side}_glove"):
                y, x, s = kp[kp_idx]
                if s > 0.2:
                    cx = int(x * frame.shape[1])
                    cy = int(y * frame.shape[0])
                    pad=15
                    cv2.rectangle(frame, (cx - pad, cy - pad), (cx + pad, cy + pad), (0, 255, 255), 2)
                    cv2.putText(frame, f"{side.capitalize()} Glove", (cx + 5, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #Final label
        glove_str = f"L-{'Yes' if glovedetected.get('left_glove') else 'No'} R-{'Yes' if glovedetected.get('right_glove') else 'No'}"
        label = f"Person {idx+1}: {punch}, {posture}, Gloves: {glove_str}"
        cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
        y_offset += line_height
    return frame

def expand_keypoints(keypoints):
    if isinstance(keypoints, str):
        try:
            keypoints = json.loads(keypoints)
        except json.JSONDecodeError:
            return pd.Series()
    if not isinstance(keypoints, list) or not all(isinstance(kp, (list, tuple)) and len(kp) == 3 for kp in keypoints):
        return pd.Series()
    try:
        data = {}
        for i, kp in enumerate(keypoints):
            data[f'x_{i}'] = kp[0]
            data[f'y_{i}'] = kp[1]
            data[f's_{i}'] = kp[2]
        return pd.Series(data)
    except Exception:
        return pd.Series()

def rescale_keypoints(keypoints, input_size, original_size):
    input_height, input_width = input_size
    orig_height, orig_width = original_size

    # Compute scale and padding from resize_with_pad
    scale = min(input_width / orig_width, input_height / orig_height)
    pad_x = (input_width - orig_width * scale) / 2
    pad_y = (input_height - orig_height * scale) / 2

    rescaled = []
    for person in keypoints:
        kp_person = []
        for y, x, s in person:
            # Undo padding and scaling
            x_unpad = (x * input_width - pad_x) / scale
            y_unpad = (y * input_height - pad_y) / scale
            kp_person.append((y_unpad / orig_height, x_unpad / orig_width, s))  # back to normalized
        rescaled.append(kp_person)
    return rescaled
def extract_bbox_from_keypoints(keypoints, threshold=0.2):
    x_coords = [kp[0] for kp in keypoints if kp[2] > threshold]
    y_coords = [kp[1] for kp in keypoints if kp[2] > threshold]
    if not x_coords or not y_coords:
        return None
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return (x_min, y_min, x_max, y_max)

def is_wearing_white(frame, bbox, white_thresh=200):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = np.sum(mask > 0) / (crop.shape[0] * crop.shape[1])
    return white_ratio > 0.3  # You can tune this
# def detect_referee(person_kpts, frame):
#     bbox = extract_bbox_from_keypoints(person_kpts)
#     return bbox and is_wearing_white(frame, bbox)

# ------------------ Detect Jersey Color ------------------

def get_jersey_color(frame, keypoints, confidence_thresh=0.3):
    keypoints = np.array(keypoints)  # Ensure it's a NumPy array

    if keypoints.shape != (17, 3):
        return "unknown1"

    h, w, _ = frame.shape
    torso_indices = [5, 6, 11, 12]  # L/R shoulder, L/R hip
    valid_points = []

    for i in torso_indices:
        y, x, conf = keypoints[i]
        if conf > confidence_thresh:
            valid_points.append((x * w, y * h))  # pixel coords

    if len(valid_points) < 2:
        return "unknown2"

    xs, ys = zip(*valid_points)
    x_min = int(np.clip(min(xs), 0, w - 1))
    x_max = int(np.clip(max(xs), 0, w - 1))
    y_min = int(np.clip(min(ys), 0, h - 1))
    y_max = int(np.clip(max(ys), 0, h - 1))

    if y_max <= y_min or x_max <= x_min:
        return "unknown3"

    cropped = frame[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return "unknown4"

    b, g, r = np.mean(cropped, axis=(0, 1))

    if r > 1.1 * b:
        return "red"
    elif b > 1.1 * r:
        return "blue"
    else:
        return "unknown5"


class SimpleIDTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = []

    def assign_ids(self, persons):
        updated_tracks = []
        for person in persons:
            found = False
            for track in self.tracks:
                if self._iou(person, track["keypoints"]) > 0.4:
                    updated_tracks.append({"id": track["id"], "keypoints": person})
                    found = True
                    break
            if not found:
                updated_tracks.append({"id": self.next_id, "keypoints": person})
                self.next_id += 1
        self.tracks = updated_tracks
        return self.tracks

    def _iou(self, k1, k2):
        # Simple IOU on shoulder-hip box
        k1 = np.array(k1)
        k2 = np.array(k2)
        try:
            x1_min = np.min(k1[[5, 6, 11, 12], 1])
            y1_min = np.min(k1[[5, 6, 11, 12], 0])
            x1_max = np.max(k1[[5, 6, 11, 12], 1])
            y1_max = np.max(k1[[5, 6, 11, 12], 0])

            x2_min = np.min(k2[[5, 6, 11, 12], 1])
            y2_min = np.min(k2[[5, 6, 11, 12], 0])
            x2_max = np.max(k2[[5, 6, 11, 12], 1])
            y2_max = np.max(k2[[5, 6, 11, 12], 0])

            xi1 = max(x1_min, x2_min)
            yi1 = max(y1_min, y2_min)
            xi2 = min(x1_max, x2_max)
            yi2 = min(y1_max, y2_max)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)

            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area else 0
        except:
            return 0
tracker = SimpleIDTracker()
# File uploader
uploaded_files = st.file_uploader("Upload boxing video", type=["mp4", "avi", "mov"], accept_multiple_files=True)
if uploaded_files:

    all_logs = []
    progress_bar = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"📦Frame Processing: {uploaded_file.name}")
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, uploaded_file.name)

        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(input_path)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        raw_output = os.path.join(temp_dir, "raw_output.mp4")
        out_writer = cv2.VideoWriter(raw_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Init before loop
        persistent_labels = {}     # {person_id: label}
        punch_log = []             # list of new punches
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        # Initialize jersey color identity map (only once)
        if 'jersey_colors_map' not in st.session_state:
            st.session_state['jersey_colors_map'] = {}

        last_punch_time = {}
        #frame loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            timestamp = frame_idx / fps

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = tf.image.resize_with_pad(tf.expand_dims(rgb_frame, axis=0), 256, 256)
            input_tensor = tf.cast(img, dtype=tf.int32)
            results = model.signatures['serving_default'](input_tensor)
            keypoints = extract_keypoints(results)
            #st.info(f"keypoints= {keypoints}")
            #st.info(f"Keypoints shape:{np.array(keypoints).shape}")

            if not keypoints:
                out_writer.write(frame)
                continue

            # Assign unique IDs to each person
            tracked = tracker.assign_ids(keypoints)

            # Display tracked player labels with jersey color
            for track in tracked:
                person_id = track["id"]
                person_kpts = track["keypoints"]
                # st.info(f"track_id= {person_id} ,track_kpts={person_kpts} ")

                # Assign stable jersey identity once
                if person_id not in st.session_state['jersey_colors_map']:
                    jersey = get_jersey_color(frame, person_kpts)
                    if jersey == "red":
                        st.session_state['jersey_colors_map'][person_id] = "redboxer"
                    elif jersey == "blue":
                        st.session_state['jersey_colors_map'][person_id] = "blueboxer"
                    else:
                        st.session_state['jersey_colors_map'][person_id] = f""

                boxer_label = st.session_state['jersey_colors_map'][person_id]
                jersey_color = get_jersey_color(frame, person_kpts)

                y, x = int(person_kpts[0][0] * h), int(person_kpts[0][1] * w)
                # cv2.putText(
                #     frame,
                #     f"{boxer_label}",
                #     (x, y - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     (0, 255, 0),
                #     2
                # )

            # # Print jersey color info
            # for i, person_kpts in enumerate(keypoints):
            #     jersey = get_jersey_color(frame, person_kpts)
            #     st.write(f"Person {i+1} jersey color: {jersey}")


            rescaledkeypoints = rescale_keypoints(keypoints, input_size=(256, 256), original_size=(height, width))  # list of keypoints for all persons in a single frame
            postures = check_posture(rescaledkeypoints)
            glove_detections=detect_gloves_by_color_and_shape(frame,rescaledkeypoints)

            h, w = frame.shape[:2]
            punches = []
            timestamp = frame_idx / fps  # timestamp in seconds


            # Initialize referee ID just once
            if 'referee_id' not in st.session_state:
                st.session_state['referee_id'] = None

            for person_id, person_kpts in enumerate(rescaledkeypoints):
                person_kpts = np.array(person_kpts)
                person_kpts[:, 0] *= width
                person_kpts[:, 1] *= height

                # Attempt to detect referee (once)
                if st.session_state['referee_id'] is None:
                    #st.info("test")
                    bbox = extract_bbox_from_keypoints(person_kpts)
                    #st.info(f"bbox ={bbox}")
                    if bbox and is_wearing_white(frame, bbox):
                        st.session_state['referee_id'] = person_id
                        # st.success(f"✅ Referee Detected (ID={person_id})")
                        continue  # Skip this frame for referee to avoid confusion

                # Skip referee in every frame after detection
                if person_id in [2, 3]:
                    continue

                label, is_new  = detect_punch(person_id, person_kpts, timestamp)
                persistent_labels[person_id] = label

                # if label != "None":
                if is_new :
                    # ✅ Normalize for jersey color extraction
                    normalized_kpts = person_kpts.copy()
                    normalized_kpts[:, 0] /= width
                    normalized_kpts[:, 1] /= height

                    color = get_jersey_color(frame, normalized_kpts)
                    punches.append({
                        "frame": frame_idx,
                        "time": round(timestamp, 2),
                        "person_id": person_id,
                        "label": label,
                        "jersey_color":color
                    })
            punch_labels = [persistent_labels.get(pid, "None") for pid in range(len(rescaledkeypoints))]
                # st.write(f"[DEBUG] referee: {st.session_state['referee_id']}, time: {round(timestamp, 2)}, label: {label}")
            # annotated = draw_annotations(frame.copy(), rescaledkeypoints, punches, postures, glove_detections, h, w)
            annotated = draw_annotations(frame.copy(), rescaledkeypoints, punch_labels, postures, glove_detections, h, w,person_id)

            out_writer.write(annotated)
            #st.text(f"Frame {frame_idx} | Punches: {punches} | rescaledkeypoints: {rescaledkeypoints}")

            for punch in punches:
                i = punch["person_id"]
                punch_log.append({
                    "video": uploaded_file.name,
                    "frame": punch["frame"],
                    # "person": i,
                    "person": st.session_state['jersey_colors_map'].get(i, f"boxer_{i}"),
                    "timestamp": punch["time"],
                    "punch": punch["label"],
                    # "jersey_color": punch["jersey_color"],
                    "posture": postures[i] if i < len(postures) else "N/A",
                    "gloves": glove_detections[i] if i < len(glove_detections) else "N/A",
                    "keypoints": keypoints[i] if i < len(keypoints) else "N/A"
                })

            frame_idx += 1
            if frame_idx % 5 == 0:
              total_progress = (idx + frame_idx / total_frames) / len(uploaded_files)
              progress_bar.progress(min(total_progress, 1.0))

        cap.release()
        out_writer.release()

        # FFmpeg encode
        final_output = os.path.join(temp_dir, f"final_{uploaded_file.name}")
        try:
          ffmpeg.input(raw_output).output(final_output, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True)
        except ffmpeg.Error as e:
          st.error("FFmpeg failed: " + str(e))

        #st.text(f"Frame {frame_idx}: {len(keypoints)} people, {len(punches)} punches")

        st.video(final_output)
        with open(final_output, "rb") as f:
            st.download_button("📥 Download Annotated Video", f, file_name=f"annotated_{uploaded_file.name}", mime="video/mp4")

        df = pd.DataFrame(punch_log)
        if df.empty:
            st.warning("⚠️ No punch data found.")
            continue

        # Speed calculation block
        df['timestamp'] = df['frame'] / fps

        # Group by video or person if needed
        df['speed (approx)'] = df.groupby('person')['timestamp'].diff().apply(lambda x: 1 / x if x and x > 0 else 0)
        # df["speed (approx)"] = (
        #     df.groupby("person")["timestamp"]
        #     .diff()
        #     .apply(lambda x: 1 / x if pd.notnull(x) and x > 0 else 0)
        # )

        # st.write("### 🔍 Keypoints Sample")
        # st.json(df['keypoints'].iloc[0])

        expanded_df = df.copy()
        keypoint_cols = df['keypoints'].apply(expand_keypoints)
        if not keypoint_cols.empty:
            expanded_df = pd.concat([df.drop(columns=['keypoints']), keypoint_cols], axis=1)
            st.dataframe(expanded_df.head())
            st.download_button("📄 Download Log CSV", expanded_df.to_csv(index=False), file_name=f"log_{uploaded_file.name}.csv", mime="text/csv")

        all_logs.extend(punch_log)
        # st.write("All columns:", expanded_df.columns.tolist())


        df_log = pd.DataFrame(punch_log)

        # Expand keypoints into flat features
        df_features = df_log['keypoints'].apply(expand_keypoints)
        df_full = pd.concat([df_log.drop(columns=['keypoints']), df_features], axis=1).dropna()

        # Extract features: all keypoints x, y, s columns
        keypoint_cols = []
        for i in range(17):
            keypoint_cols.extend([f'x_{i}', f'y_{i}', f's_{i}'])

        # st.write("df_full columns:", df_full.columns.tolist())
        # st.info(f"All keypoint_cols  in dataframe: {keypoint_cols}")


        # Add index to track row
        df_full = df_full.reset_index(drop=False)  # 'index' column will store original indices

        # Then extract features and target
        X = df_full[keypoint_cols].values
        y = df_full["punch"]
        indices = df_full["index"]  # This holds row index from original DataFrame


        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        joblib.dump(clf, "punch_classifier_model.joblib")

        with open("punch_classifier_model.joblib", "rb") as f:
          st.download_button(
              label="📥 Download Trained Classifier",
              data=f,
              file_name="punch_classifier_model.joblib",
              mime="application/octet-stream"
          )

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        #st.info(f" Accuracy:, {acc}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

        # Reconstruct predictions DataFrame with correct alignment
        true_labels = pd.Series(y_test).reset_index(drop=True)
        pred_labels = pd.Series(y_pred).reset_index(drop=True)

        st.write("Punch label counts:\n", y.value_counts())
        #Only keep existing metadata columns
        meta_columns = ["video", "frame", "person", "timestamp", "speed (approx)"]
        meta_columns = [col for col in meta_columns if col in expanded_df.columns]

        pred_meta = expanded_df.loc[idx_test][meta_columns].reset_index(drop=True)

        pred_output_df = pd.concat([
            pred_meta,
            true_labels.rename("true_label"),
            pred_labels.rename("predicted_label")
        ], axis=1)

        st.dataframe(pred_output_df.head())

        st.download_button(
            "📄 Download Predictions CSV",
            pred_output_df.to_csv(index=False),
            file_name="predictions_vs_actual.csv",
            mime="text/csv"
        )

        # Heatmap
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        # Detailed Report
        #st.info(f"\n Classification Report:\n= {classification_report(y_test, y_pred)}")

        # === Performance Metrics Summary ===

        # Count the number of each predicted label
        st.subheader("🍩 Punch Count (Pie Chart)")
        label_counts = expanded_df['punch'].value_counts()

        # Plot pie chart
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            label_counts.values,
            labels=label_counts.index,
            autopct='%1.1f%%',
            startangle=90
        )
        ax.axis('equal')  # Equal aspect ratio for a circle

        # Display in Streamlit
        st.pyplot(fig)

        # Accuracy display
        #st.metric("✅ Accuracy", f"{acc:.2%}")

        # st.subheader("📊 Per-Punch Speed Over Time (Bar Chart)")

        # # Ensure timestamp is in seconds
        # pred_output_df["timestamp_sec"] = pred_output_df["timestamp"].astype(float)

        # st.bar_chart(
        #     pred_output_df.set_index("timestamp_sec")["speed (approx)"],
        #     height=300,
        #     use_container_width=True
        # )

        # st.subheader("📈 Per-Punch Speed Over Time")

        # # Ensure timestamp is in seconds
        # pred_output_df["timestamp_sec"] = pred_output_df["timestamp"].astype(float)

        # # Plot punch speed
        # st.line_chart(
        #     pred_output_df.set_index("timestamp_sec")["speed (approx)"],
        #     height=300,
        #     use_container_width=True
        # )

        # Punch Counts
        st.subheader("📊 Punch Type Distribution")
        punch_counts =expanded_df['punch'].value_counts()
        st.bar_chart(punch_counts)

        #st.subheader("📊 Punch Type Distribution2")
        # Replace df with filtered_df in all groupby, charts, etc.
        df_to_use = expanded_df  # instead of pred_output_df

        # Punch frequency over time

        st.subheader("📉 Punch Frequency Over Time")
        # Round timestamps to 1 second
        pred_output_df["time_bin"] = expanded_df["timestamp"].round(0)

        # Count punches per second
        time_grouped = expanded_df[expanded_df["punch"].notna()] \
            .groupby(["timestamp", "punch"]).size().unstack().fillna(0)

        #st.line_chart(time_grouped)

        import altair as alt
        melted = time_grouped.reset_index().melt('timestamp', var_name='Punch', value_name='Count')

        chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X("timestamp:O", title="Time (s)"),
            y=alt.Y("Count:Q", title="Punch Count"),
            color="Punch:N",
            tooltip=["timestamp", "Punch", "Count"]
        ).properties(height=300)

        st.altair_chart(chart, use_container_width=True)

        # Filter for punches
        valid_punches = expanded_df[expanded_df["punch"].notna()]

        # Compute punches
        total_punches = len(valid_punches)

        # Use full timestamp range, NOT just punch timestamps!
        start_time = expanded_df["timestamp"].min()
        end_time = expanded_df["timestamp"].max()
        duration = end_time - start_time

        # # Debug print (optional)
        # st.write(f"Start Time: {start_time}, End Time: {end_time}, Duration: {duration:.2f}s, Punches: {total_punches}")

        # # Final punch speed
        # punch_speed = total_punches / duration if duration > 0 else 0
        # st.metric("⚡ Average Punch Speed (approx)", f"{punch_speed:.2f} punches/sec")

        # # 🎯 Map person_id to jersey color
        # color_map = st.session_state.get("jersey_colors_map", {})
        # expanded_df["boxer"] = expanded_df["person"].map(color_map).fillna("unknown")

        # # 👥 Count punches per boxer (red/blue)
        # st.subheader("👥 Punch Count per Boxer")
        # boxer_punch_counts = expanded_df.groupby("boxer")["punch"].value_counts().unstack().fillna(0)
        # st.dataframe(boxer_punch_counts)

        # Count by Person
        st.subheader("👥 Punch Count per Person")
        person_punch_counts = expanded_df.groupby("person")["punch"].value_counts().unstack().fillna(0)
        st.dataframe(person_punch_counts)

        # Confusion matrix chart (if not shown already)
        st.subheader("🔁 Confusion Matrix")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues", ax=ax2)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        st.pyplot(fig2)

        # Classification report
        # st.subheader("📋 Classification Report")
        # report_str = classification_report(y_test, y_pred, output_dict=False)
        # st.text(report_str)

    progress_bar.empty()

requirements = '''streamlit
numpy>=1.21.0
scikit-learn>=1.0
tensorflow
tensorflow_hub
opencv-python-headless
pandas
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
