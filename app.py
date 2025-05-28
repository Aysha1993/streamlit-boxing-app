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
#import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


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


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def detect_punch(keypoints):
    #st.info(f"kp={keypoints}")
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    NOSE = 0
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    lw = keypoints[LEFT_WRIST][:2]
    rw = keypoints[RIGHT_WRIST][:2]
    nose = keypoints[NOSE][:2]
    le = keypoints[LEFT_ELBOW][:2]
    re = keypoints[RIGHT_ELBOW][:2]
    ls = keypoints[LEFT_SHOULDER][:2]
    rs = keypoints[RIGHT_SHOULDER][:2]

    # Distances from wrists to nose (used for punches)
    dist_lw_nose = np.linalg.norm(lw - nose)
    dist_rw_nose = np.linalg.norm(rw - nose)

    # Elbow angles to check punch extension
    left_elbow_angle = calculate_angle(ls, le, lw)
    right_elbow_angle = calculate_angle(rs, re, rw)

    # Face position to estimate duck
    head_height = nose[1]

    # Heuristics
    if dist_lw_nose > 50 and left_elbow_angle > 130:
        return "Jab"
    elif dist_rw_nose > 50 and right_elbow_angle > 130:
        return "Cross"
    elif dist_lw_nose < 50 and dist_rw_nose < 50:
        return "Guard"
    elif head_height > rs[1] + 40 and head_height > ls[1] + 40:
        return "Duck"
    else:
        return "None"


# def classify_punch(keypoints_all_people, frame_idx):
#     global person_states
#     results = []

#     for person_id, kpts in enumerate(keypoints_all_people):
#         # Initialize or retrieve previous state
#         state = person_states.get(person_id, {
#             "prev_kpts": kpts,
#             "in_motion": {"left": False, "right": False},
#             "frame_start": {"left": None, "right": None},
#         })

#         for side in ["left", "right"]:
#             try:
#                 wrist_idx = keypoint_index[f"{side}_wrist"]
#                 elbow_idx = keypoint_index[f"{side}_elbow"]
#                 shoulder_idx = keypoint_index[f"{side}_shoulder"]

#                 wrist = kpts[wrist_idx][:2]
#                 elbow = kpts[elbow_idx][:2]
#                 shoulder = kpts[shoulder_idx][:2]
#                 prev_wrist = state["prev_kpts"][wrist_idx][:2]

#                 # Sanity check for missing data
#                 if not all(map(lambda x: isinstance(x, (int, float)), wrist + elbow + shoulder + prev_wrist)):
#                     continue

#                 velocity = calculate_velocity(prev_wrist, wrist)
#                 elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)

#                 # Start of punch
#                 if velocity > VELOCITY_THRESHOLD and not state["in_motion"][side]:
#                     state["in_motion"][side] = True
#                     state["frame_start"][side] = frame_idx

#                 # End of punch
#                 elif velocity < VELOCITY_THRESHOLD * 0.5 and state["in_motion"][side]:
#                     frame_start = state["frame_start"][side]
#                     frame_end = frame_idx
#                     punch_type = None

#                     # Classify punch type based on elbow angle
#                     if elbow_angle < HOOK_ANGLE_THRESHOLD:
#                         punch_type = "Hook"
#                     elif side == "left":
#                         punch_type = "Jab"
#                     else:
#                         punch_type = "Cross"

#                     if punch_type:
#                         results.append({
#                             "label": f"{side.capitalize()} {punch_type}",
#                             "frame_start": frame_start,
#                             "frame_end": frame_end
#                         })

#                     # Reset motion state
#                     state["in_motion"][side] = False
#                     state["frame_start"][side] = None

#             except KeyError as e:
#                 print(f"Missing keypoint index: {e}")
#                 continue
#             except IndexError as e:
#                 print(f"Keypoint index out of range: {e}")
#                 continue
#             except Exception as e:
#                 print(f"Unexpected error in punch detection: {e}")
#                 continue

#         # Update previous keypoints for next frame
#         state["prev_kpts"] = kpts
#         person_states[person_id] = state

#     return results

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

import cv2
import numpy as np

def detect_gloves_by_color_and_shape(frame, keypoints, confidence_threshold=0.3, crop_size=30):
    """
    Detect boxing gloves using wrist keypoints and color/shape around the wrist.

    Args:
        frame: BGR image (numpy array)
        keypoints: List of people, each a list of 17 keypoints (x, y, confidence)
        confidence_threshold: Minimum confidence for wrist visibility
        crop_size: Half-size of square patch to crop around wrist

    Returns:
        List of glove detections: [{'left_glove': True/False, 'right_glove': True/False}, ...]
    """
    glove_detections = []

    for person in keypoints:
        h, w, _ = frame.shape

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

            # Convert to HSV for better color filtering
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

            # Define color ranges (adjust as needed)
            glove_colors = {
                'red': ((0, 70, 50), (10, 255, 255)),
                'blue': ((100, 100, 50), (140, 255, 255)),
                'black': ((0, 0, 0), (180, 255, 60)),
            }

            mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in glove_colors.values():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask_total = cv2.bitwise_or(mask_total, mask)

            # Check if mask has enough coverage (glove likely present)
            glove_ratio = np.sum(mask_total > 0) / mask_total.size
            return glove_ratio > 0.2  # You can tune this threshold

        left_crop = crop_wrist_region(9)
        right_crop = crop_wrist_region(10)

        left_glove = is_glove(left_crop)
        right_glove = is_glove(right_crop)

        glove_detections.append({
            'left_glove': left_glove,
            'right_glove': right_glove
        })

    return glove_detections

# def detect_gloves(keypoints, distance_thresh=0.1):
#     gloves = []
#     for kp in keypoints:
#         lw, le = kp[9], kp[7]   # left wrist, left elbow
#         rw, re = kp[10], kp[8]  # right wrist, right elbow

#         def is_glove_present(wrist, elbow):
#             if wrist[2] > 0.2 and elbow[2] > 0.2:
#                 dist = np.linalg.norm(np.array(wrist[:2]) - np.array(elbow[:2]))
#                 return dist > distance_thresh
#             return False

#         gloves.append({
#             "left": is_glove_present(lw, le),
#             "right": is_glove_present(rw, re)
#         })

#     return gloves

# 17 keypoints (based on MoveNet/COCO order)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton edges between keypoints
SKELETON_EDGES = [
    (0, 1), (1, 3), (0, 2), (2, 4),         # Face
    (5, 7), (7, 9), (6, 8), (8, 10),        # Arms
    (5, 6), (5, 11), (6, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16), # Legs
    (11, 12)                                # Hip line
]


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

import cv2

def draw_annotations(frame, keypoints, punches, postures, glove_detections, h, w):
    y_offset = 30
    line_height = 20

    valid_detections = []
    for idx, (kp_raw, punch, posture, glovedetected) in enumerate(zip(keypoints, punches, postures, glove_detections)):
        kp = np.array(kp_raw).reshape(-1, 3).tolist()
        #kp_norm = [[y / h, x / w, s] for y, x, s in kp]

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

        # for side, wrist_idx in zip(["L", "R"], [9, 10]):
        #     y, x, s = kp[wrist_idx]
        #     if s > 0.2:
        #         cx, cy = int(x * w), int(y * h)
        #         pad = 15
        #         has_glove = glove.get('left' if side == 'L' else 'right', False)
        #         color = (0, 0, 255) if has_glove else (0, 255, 255)
        #         cv2.rectangle(frame, (cx - pad, cy - pad), (cx + pad, cy + pad), color, 2)
        #         cv2.putText(frame, f"{side} Glove", (cx - pad, cy - pad - 5),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        #Draw gloves
        for side, kp_idx in [('left', 9), ('right', 10)]:
            if glovedetected.get(f"{side}_glove"):
                y, x, s = kp[kp_idx]
                if s > 0.2:
                    cx = int(x * frame.shape[1])
                    cy = int(y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.putText(frame, f"{side.capitalize()} Glove", (cx + 5, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #Final label
        glove_str = f"L-{'Yes' if glovedetected.get('left') else 'No'} R-{'Yes' if glovedetected.get('right') else 'No'}"
        label = f"Person {idx+1}: {punch}, {posture}, Gloves: {glove_str}"
        cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)
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

# File uploader
uploaded_files = st.file_uploader("Upload  boxing video", type=["mp4", "avi", "mov"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    #model = tf.saved_model.load("PATH_TO_YOUR_MOVENET_MODEL")  # Preload model once

    all_logs = []
    progress_bar = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"📦 Processing: {uploaded_file.name}")
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # resized = cv2.resize(frame, (256, 256))
            # input_tensor = tf.convert_to_tensor(resized[None, ...], dtype=tf.int32)
            # results = model.signatures['serving_default'](input_tensor)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = tf.image.resize_with_pad(tf.expand_dims(rgb_frame, axis=0), 256, 256)
            input_tensor = tf.cast(img, dtype=tf.int32)
            results = model.signatures['serving_default'](input_tensor)
            keypoints = extract_keypoints(results)
            # 🚫 Filter out likely coaches
            #keypoints = [kp for kp in keypoints if not is_likely_coach(kp)]

            if not keypoints:
                out_writer.write(frame)
                continue
            rescaledkeypoints = rescale_keypoints(keypoints, input_size=(256, 256), original_size=(height, width))
            #st.info(f"rescaledkp={rescaledkeypoints}")
            #punches = classify_punch(rescaledkeypoints,frame_idx)
            #punches = detect_punch(rescaledkeypoints)
            postures = check_posture(rescaledkeypoints)
            #gloves = detect_gloves(rescaledkeypoints)
            glove_detections=detect_gloves_by_color_and_shape(frame,rescaledkeypoints)

            h, w = frame.shape[:2]

            punches = []
            for person_kpts in rescaledkeypoints:
                person_kpts = np.array(person_kpts)  # Shape: (17, 3)
                person_kpts[:, 0] *= width  # x-coordinate
                person_kpts[:, 1] *= height  # y-coordinate

                label = detect_punch(person_kpts)
                punches.append(label)

            #annotated = draw_annotations(frame.copy(), rescaledkeypoints, punches, postures, gloves)
            annotated = draw_annotations(frame.copy(), rescaledkeypoints, punches, postures, glove_detections, h, w)

            out_writer.write(annotated)

            # for i in range(len(punches)):
            #   punch_label = punches[i]["label"] if punches[i] else "None"
            #   frame_start = punches[i]["frame_start"] if punches[i] else None
            #   frame_end = punches[i]["frame_end"] if punches[i] else None
            #   punch_log.append({
            #       "video": uploaded_file.name,
            #       "frame": frame_idx,
            #       "person": i,
            #       "punch": punch_label,
            #       "frame_start": frame_start,
            #       "frame_end": frame_end,
            #       "posture": postures[i] if i < len(postures) else "N/A",
            #       "gloves": gloves[i] if i < len(gloves) else "N/A",
            #       "keypoints": keypoints[i] if i < len(keypoints) else "N/A"
            #   })
            # st.info(f"punches = {punches}")
            for i in range(len(punches)):
                punch_log.append({
                      "video": uploaded_file.name,
                      "frame": frame_idx,
                      "person": i,
                      "timestamp": frame_idx / fps,
                      "punch": punches[i] if i < len(punches) else "N/A",
                      "posture": postures[i] if i < len(postures) else "N/A",
                      "gloves": gloves[i] if i < len(gloves) else "N/A",
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

        #ffmpeg.input(raw_output).output(final_output, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True)
        st.text(f"Frame {frame_idx}: {len(keypoints)} people, {len(punches)} punches")

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

        st.write("### 🔍 Keypoints Sample")
        st.json(df['keypoints'].iloc[0])

        expanded_df = df.copy()
        keypoint_cols = df['keypoints'].apply(expand_keypoints)
        if not keypoint_cols.empty:
            expanded_df = pd.concat([df.drop(columns=['keypoints']), keypoint_cols], axis=1)
            st.dataframe(expanded_df.head())
            st.download_button("📄 Download Log CSV", expanded_df.to_csv(index=False), file_name=f"log_{uploaded_file.name}.csv", mime="text/csv")

        all_logs.extend(punch_log)

        # Load data (assuming it's saved as CSV)

        # Flatten punch_log to DataFrame
        df_log = pd.DataFrame(punch_log)

        # Expand keypoints into flat features
        df_features = df_log['keypoints'].apply(expand_keypoints)
        df_full = pd.concat([df_log.drop(columns=['keypoints']), df_features], axis=1).dropna()


        # Drop rows where punch is missing or N/A
        df_full = df_full[df_full['punch'].notna()]
        df_full = df_full[df_full['punch'] != 'N/A']

        # df.columns = df.columns.str.strip()

        # Extract features: all keypoints x, y, s columns
        keypoint_cols = []
        for i in range(17):
            keypoint_cols.extend([f'x_{i}', f'y_{i}', f's_{i}'])

        st.write("DataFrame columns:", df_full.columns.tolist())

        # print("All keypoint columns in dataframe:", all(col in df.columns for col in keypoint_cols))  # Should be True
        st.info(f"All keypoint columns in dataframe: {keypoint_cols}")
        st.info(f"Frame: {frame_idx} | Timestamp: {frame_idx / fps:.2f} sec | Punches: {punches}")

        # Encode labels
        le = LabelEncoder()

        # Extract features and target
        X = df_full[keypoint_cols].values # Replace with actual feature column names
        y = le.fit_transform(df_full['punch'])

        # Ensure DataFrame index is clean
        df_full = df_full.reset_index(drop=True)

        # Track indices during train-test split
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, df_full.index, test_size=0.2, stratify=y, random_state=42
        )

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        # Train classifier
        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
        clf.fit(X_train_balanced, y_train_balanced)

        # Predict
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_)

        # Streamlit metrics
        st.success(f"✅ RF Accuracy (SMOTE + weights): {accuracy:.3f}")
        st.text("🔍 Classification Report:\n" + report)

        # ✅ Create predicted punch dataframe using tracked indices
        predicted_df = pd.DataFrame({
            'frame': df_full.loc[test_idx, 'frame'].values,
            'timestamp': df_full.loc[test_idx, 'frame'].values / fps,
            'predicted_label': y_pred,
            'pred_punch_type': le.inverse_transform(y_pred),
            'true_label': y_test,
            'true_punch_type': le.inverse_transform(y_test)
        })

        # Reorder and save
        predicted_df = predicted_df[['frame', 'timestamp', 'pred_punch_type', 'predicted_label', 'true_punch_type', 'true_label']]
        predicted_df.to_csv("predicted_punches.csv", index=False)

        # Streamlit display and download
        st.success("✅ Saved predicted punches to predicted_punches.csv")
        st.dataframe(predicted_df.head())
        st.download_button(
            "📄 Download Pred Log CSV",
            predicted_df.to_csv(index=False),
            file_name=f"log_{uploaded_file.name}.csv",
            mime="text/csv"
        )


        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import plotly.express as px

        # --- Section: Metrics Summary ---
        st.subheader("📈 Model Performance Metrics")
        st.metric("Accuracy", f"{accuracy:.2%}")

        # --- Section: Confusion Matrix ---
        st.subheader("🔀 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm)

        # --- Section: Punch Type Counts ---
        st.subheader("👊 Punch Type Distribution")

        # Count of predicted and true punch types
        true_counts = predicted_df['true_punch_type'].value_counts().rename("True Count")
        pred_counts = predicted_df['pred_punch_type'].value_counts().rename("Predicted Count")
        count_df = pd.concat([true_counts, pred_counts], axis=1).fillna(0).astype(int)
        st.dataframe(count_df)

        # --- Bar chart for punch frequency ---
        fig_bar = px.bar(
            count_df.reset_index(),
            x='index',
            y=['True Count', 'Predicted Count'],
            barmode='group',
            labels={'index': 'Punch Type'},
            title="🔢 Punch Frequency (True vs Predicted)"
        )
        st.plotly_chart(fig_bar)

        # --- Pie chart for predicted punches ---
        fig_pie = px.pie(
            predicted_df,
            names='pred_punch_type',
            title="🥧 Punch Prediction Breakdown"
        )
        st.plotly_chart(fig_pie)

        # --- Section: Speed Approximation ---
        st.subheader("💨 Punch Speed Estimation (approx)")

        # Approximate speed = 1 / time between punches (frames with different punches)
        predicted_df_sorted = predicted_df.sort_values(by='frame')
        predicted_df_sorted['frame_diff'] = predicted_df_sorted['frame'].diff().fillna(0)
        predicted_df_sorted['time_diff'] = predicted_df_sorted['frame_diff'] / fps
        predicted_df_sorted['approx_speed'] = 1 / predicted_df_sorted['time_diff'].replace(0, float('nan'))

        # Line chart of speed
        fig_line = px.line(
            predicted_df_sorted,
            x='frame',
            y='approx_speed',
            title='📉 Approximate Punch Speed over Time',
            labels={'approx_speed': 'Speed (punches/sec)'}
        )
        st.plotly_chart(fig_line)

        # Optional: Show raw predicted_df again
        with st.expander("🧾 View Full Predictions Table"):
            st.dataframe(predicted_df)





        # # Get the original indices of the test set
        # X_test_indices = X_test.index if isinstance(X_test, pd.DataFrame) else df_full.iloc[X_test].index

        # # Create predicted punch dataframe
        # predicted_df = pd.DataFrame({
        #     'frame': df_full.loc[X_test_indices, 'frame'].values,
        #     'timestamp': df_full.loc[X_test_indices, 'frame'].values / fps,
        #     'predicted_label': y_pred,
        #     'punch_type': le.inverse_transform(y_pred)
        # })

        # # Optional: include ground truth for comparison
        # predicted_df['true_label'] = y_test
        # predicted_df['true_punch_type'] = le.inverse_transform(y_test)

        # # Reorder and save
        # predicted_df = predicted_df[['frame', 'timestamp', 'punch_type', 'predicted_label','true_punch_type','true_label']]
        # predicted_df.to_csv("predicted_punches.csv", index=False)
        # st.success("✅ Saved predicted punches to predicted_punches.csv")
        # st.dataframe(predicted_df.head())
        # st.download_button("📄 Download Pred Log CSV", predicted_df.to_csv(index=False), file_name=f"log_{uploaded_file.name}.csv", mime="text/csv")




        # # Encode labels
        # le = LabelEncoder()
        # y = le.fit_transform(df_full['punch'].values)

        # # Split train-test
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=42)

        # # Optional: scale features
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        # # Train a Random Forest Classifier
        # clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        # clf.fit(X_train, y_train)

        # # Predict on test set
        # y_pred = clf.predict(X_test)

        # # Evaluation
        # st.info(f"RF Accuracy with scaler:  {accuracy_score(y_test, y_pred)}")
        # classification=classification_report(y_test, y_pred, target_names=le.classes_)
        # st.info(f" classification = {classification}")


        # # Expand keypoints into flat features
        # df_features = df_log['keypoints'].apply(expand_keypoints)
        # df_full = pd.concat([df_log.drop(columns=['keypoints']), df_features], axis=1).dropna()

        # st.success("✅ Extracted keypoints and labels for ML training.")


        # # Label encode target
        # label_encoder = LabelEncoder()
        # df_full['label'] = label_encoder.fit_transform(df_full['punch'])

        # # Feature/target split
        # X = df_full[[col for col in df_full.columns if col.startswith(('x_', 'y_', 's_'))]]
        # y = df_full['label']

        # # Train/test split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # # Train classifiers
        # svm_model = svm.SVC(kernel='rbf')
        # tree_model = DecisionTreeClassifier(max_depth=5)

        # svm_model.fit(X_train, y_train)
        # tree_model.fit(X_train, y_train)

        # # Evaluate
        # y_pred_svm = svm_model.predict(X_test)
        # y_pred_tree = tree_model.predict(X_test)

        # acc_svm = accuracy_score(y_test, y_pred_svm)
        # acc_tree = accuracy_score(y_test, y_pred_tree)

        # st.subheader("📈 Model Evaluation")
        # st.write(f"🔹 SVM Accuracy: `{acc_svm:.2f}`")
        # st.write(f"🔹 Decision Tree Accuracy: `{acc_tree:.2f}`")

        # # Confusion Matrix
        # st.write("### Confusion Matrix (SVM)")
        # cm = confusion_matrix(y_test, y_pred_svm)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        # fig, ax = plt.subplots(figsize=(6, 4))
        # disp.plot(ax=ax, cmap='Blues')
        # st.pyplot(fig)

        # st.subheader("🎬 Visualize Predictions")

        # # Run classifier on a few frames
        # sample_preds = []
        # for i in range(min(10, len(X_test))):
        #     pred_label = label_encoder.inverse_transform([svm_model.predict([X_test.iloc[i]])[0]])[0]
        #     actual_label = label_encoder.inverse_transform([y_test.iloc[i]])[0]
        #     sample_preds.append(f"✅ Predicted: {pred_label} | 🏷️ Actual: {actual_label}")

        # for row in sample_preds:
        #     st.write(row)
        # st.markdown("## 🥊 Punch Performance Dashboard")

        # if 'punch_log' in locals() and len(punch_log) > 0:
        #     df = pd.DataFrame(punch_log)

        #     # Count Punch Types
        #     type_counts = df['punch'].value_counts().to_dict()
        #     st.subheader("🔢 Punch Type Count")
        #     cols = st.columns(len(type_counts))
        #     for i, (ptype, count) in enumerate(type_counts.items()):
        #         cols[i].metric(label=ptype, value=count)

        #     # Approximate Punch Frequency
        #     if 'frame_end' in df.columns:
        #         duration_frames = df['frame_end'].max() - df['frame_start'].min()
        #         fps = 30  # adjust this to your actual FPS
        #         duration_sec = duration_frames / fps if fps else 1
        #         punch_speed = len(df) / duration_sec if duration_sec > 0 else 0
        #         st.subheader("⚡ Speed Approximation")
        #         st.metric("Punches per Second", f"{punch_speed:.2f}")
        #     else:
        #         st.warning("Frame timing info missing — can't compute speed.")

        #     # Time-bucketed Frequency Chart
        #     if 'frame_start' in df.columns:
        #         df['time_sec'] = df['frame_start'] // 30  # adjust for your FPS
        #         time_counts = df.groupby('time_sec')['punch'].count()
        #         st.subheader("📈 Punch Frequency Over Time")
        #         fig1, ax1 = plt.subplots()
        #         time_counts.plot(kind='line', marker='o', ax=ax1)
        #         ax1.set_xlabel("Time (s)")
        #         ax1.set_ylabel("Punches")
        #         ax1.set_title("Punches Per Second")
        #         st.pyplot(fig1)

        #     # Bar Chart of Punch Types
        #     st.subheader("📊 Punch Type Distribution")
        #     fig2, ax2 = plt.subplots()
        #     ax2.bar(type_counts.keys(), type_counts.values(), color='skyblue')
        #     #sns.barplot(x=list(type_counts.keys()), y=list(type_counts.values()), ax=ax2)
        #     ax2.set_ylabel("Count")
        #     ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        #     st.pyplot(fig2)

        #     # Pie Chart of Punch Types
        #     st.subheader("🥧 Punch Share - Pie Chart")
        #     fig3, ax3 = plt.subplots()
        #     ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
        #     ax3.axis('equal')
        #     st.pyplot(fig3)
        # else:
        #     st.info("🔍 No punch data found. Upload and process a video to see metrics.")

    progress_bar.empty()


requirements = '''streamlit
tensorflow
tensorflow_hub
opencv-python-headless
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
