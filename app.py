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
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
#import seaborn as sns


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

def classify_punch(keypoints_all_people, frame_idx):
    global person_states
    results = []

    for person_id, kpts in enumerate(keypoints_all_people):
        # Initialize or retrieve previous state
        state = person_states.get(person_id, {
            "prev_kpts": kpts,
            "in_motion": {"left": False, "right": False},
            "frame_start": {"left": None, "right": None},
        })

        for side in ["left", "right"]:
            try:
                wrist_idx = keypoint_index[f"{side}_wrist"]
                elbow_idx = keypoint_index[f"{side}_elbow"]
                shoulder_idx = keypoint_index[f"{side}_shoulder"]

                wrist = kpts[wrist_idx][:2]
                elbow = kpts[elbow_idx][:2]
                shoulder = kpts[shoulder_idx][:2]
                prev_wrist = state["prev_kpts"][wrist_idx][:2]

                # Sanity check for missing data
                if not all(map(lambda x: isinstance(x, (int, float)), wrist + elbow + shoulder + prev_wrist)):
                    continue

                velocity = calculate_velocity(prev_wrist, wrist)
                elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)

                # Start of punch
                if velocity > VELOCITY_THRESHOLD and not state["in_motion"][side]:
                    state["in_motion"][side] = True
                    state["frame_start"][side] = frame_idx

                # End of punch
                elif velocity < VELOCITY_THRESHOLD * 0.5 and state["in_motion"][side]:
                    frame_start = state["frame_start"][side]
                    frame_end = frame_idx
                    punch_type = None

                    # Classify punch type based on elbow angle
                    if elbow_angle < HOOK_ANGLE_THRESHOLD:
                        punch_type = "Hook"
                    elif side == "left":
                        punch_type = "Jab"
                    else:
                        punch_type = "Cross"

                    if punch_type:
                        results.append({
                            "label": f"{side.capitalize()} {punch_type}",
                            "frame_start": frame_start,
                            "frame_end": frame_end
                        })

                    # Reset motion state
                    state["in_motion"][side] = False
                    state["frame_start"][side] = None

            except KeyError as e:
                print(f"Missing keypoint index: {e}")
                continue
            except IndexError as e:
                print(f"Keypoint index out of range: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error in punch detection: {e}")
                continue

        # Update previous keypoints for next frame
        state["prev_kpts"] = kpts
        person_states[person_id] = state

    return results

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

# def draw_annotations(frame, keypoints_with_scores, threshold=0.2):
#     h, w, _ = frame.shape

#     for person in keypoints_with_scores:
#         keypoints = person[:17]

#         # Draw keypoints with names
#         for i, (y, x, score) in enumerate(keypoints):  # (y, x, score)
#             cx = int(x * w)
#             cy = int(y * h)
#             color = (0, 255, 255) if score > threshold else (255, 0, 255)
#             cv2.circle(frame, (cx, cy), 4, color, -1)
#             cv2.putText(frame, KEYPOINT_NAMES[i], (cx + 5, cy - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

#         # Draw skeleton connections
#         for p1, p2 in SKELETON_EDGES:
#             y1, x1, s1 = keypoints[p1]
#             y2, x2, s2 = keypoints[p2]
#             if s1 > threshold and s2 > threshold:
#                 pt1 = (int(x1 * w), int(y1 * h))
#                 pt2 = (int(x2 * w), int(y2 * h))
#                 cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

#     return frame


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


def draw_annotations(frame, keypoints, punches, postures, gloves):
    h, w = frame.shape[:2]
    print("keypoints:", len(keypoints), "punches:", len(punches), "postures:", len(postures), "gloves:", len(gloves))
    

    max_people = len(keypoints)
    punches = punches + [""] * (max_people - len(punches))
    postures = postures + [""] * (max_people - len(postures))
    gloves = gloves + [""] * (max_people - len(gloves))

    y_offset = 30
    line_height = 20

    for idx, (kp, punch, posture, glove) in enumerate(zip(keypoints, punches, postures, gloves)):
        # Draw keypoints
        for i, (y, x, s) in enumerate(kp):
            if s > 0.2:
                cx, cy = int(x * w), int(y * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(frame, KEYPOINT_NAMES[i], (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw skeleton
        for (p1, p2) in SKELETON_EDGES:
            y1, x1, s1 = kp[p1]
            y2, x2, s2 = kp[p2]
            if s1 > 0.2 and s2 > 0.2:
                pt1 = int(x1 * w), int(y1 * h)
                pt2 = int(x2 * w), int(y2 * h)
                if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Draw gloves only if wrist confidence is high enough
        for side, wrist_idx in zip(["L", "R"], [9, 10]):
            y, x, s = kp[wrist_idx]
            st.info(f"{side} wrist score: {s:.2f} at x={x:.2f}, y={y:.2f}")
            if s > 0.65:  # tightened from 0.5 to 0.65
                cx, cy = int(x * w), int(y * h)
                if 0 <= cx < w and 0 <= cy < h:
                    pad = 15
                    cv2.rectangle(frame, (cx - pad, cy - pad), (cx + pad, cy + pad), (0, 0, 255), 2)
                    cv2.putText(frame, f"{side} Glove", (cx - pad, cy - pad - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.putText(frame, f"{s:.2f}", (cx + 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Punch/Posture label
        left_label = f"Person {idx+1}: {punch}, {posture}"
        cv2.putText(frame, left_label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
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
            rescaledkeypoints = rescale_keypoints(keypoints, input_size=(256, 256), original_size=(height, width))
            if not keypoints:
                out_writer.write(frame)
                continue

            punches = classify_punch(rescaledkeypoints,frame_idx)
            postures = check_posture(rescaledkeypoints)
            gloves = detect_gloves(rescaledkeypoints)

            h, w = frame.shape[:2]
            
            
            annotated = draw_annotations(frame.copy(), rescaledkeypoints, punches, postures, gloves)
            
            out_writer.write(annotated)

            for i in range(len(punches)):
              punch_label = punches[i]["label"] if punches[i] else "None"
              frame_start = punches[i]["frame_start"] if punches[i] else None
              frame_end = punches[i]["frame_end"] if punches[i] else None
              punch_log.append({
                  "video": uploaded_file.name,
                  "frame": frame_idx,
                  "person": i,
                  "punch": punch_label,
                  "frame_start": frame_start,
                  "frame_end": frame_end,
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


        # Flatten punch_log to DataFrame
        df_log = pd.DataFrame(punch_log)

        # Expand keypoints into flat features
        df_features = df_log['keypoints'].apply(expand_keypoints)
        df_full = pd.concat([df_log.drop(columns=['keypoints']), df_features], axis=1).dropna()

        st.success("✅ Extracted keypoints and labels for ML training.")


        # Label encode target
        label_encoder = LabelEncoder()
        df_full['label'] = label_encoder.fit_transform(df_full['punch'])

        # Feature/target split
        X = df_full[[col for col in df_full.columns if col.startswith(('x_', 'y_', 's_'))]]
        y = df_full['label']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Train classifiers
        svm_model = svm.SVC(kernel='rbf')
        tree_model = DecisionTreeClassifier(max_depth=5)

        svm_model.fit(X_train, y_train)
        tree_model.fit(X_train, y_train)

        # Evaluate
        y_pred_svm = svm_model.predict(X_test)
        y_pred_tree = tree_model.predict(X_test)

        acc_svm = accuracy_score(y_test, y_pred_svm)
        acc_tree = accuracy_score(y_test, y_pred_tree)

        st.subheader("📈 Model Evaluation")
        st.write(f"🔹 SVM Accuracy: `{acc_svm:.2f}`")
        st.write(f"🔹 Decision Tree Accuracy: `{acc_tree:.2f}`")

        # Confusion Matrix
        st.write("### Confusion Matrix (SVM)")
        cm = confusion_matrix(y_test, y_pred_svm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        fig, ax = plt.subplots(figsize=(6, 4))
        disp.plot(ax=ax, cmap='Blues')
        st.pyplot(fig)

        st.subheader("🎬 Visualize Predictions")

        # Run classifier on a few frames
        sample_preds = []
        for i in range(min(10, len(X_test))):
            pred_label = label_encoder.inverse_transform([svm_model.predict([X_test.iloc[i]])[0]])[0]
            actual_label = label_encoder.inverse_transform([y_test.iloc[i]])[0]
            sample_preds.append(f"✅ Predicted: {pred_label} | 🏷️ Actual: {actual_label}")

        for row in sample_preds:
            st.write(row)
        st.markdown("## 🥊 Punch Performance Dashboard")

        if 'punch_log' in locals() and len(punch_log) > 0:
            df = pd.DataFrame(punch_log)

            # Count Punch Types
            type_counts = df['punch'].value_counts().to_dict()
            st.subheader("🔢 Punch Type Count")
            cols = st.columns(len(type_counts))
            for i, (ptype, count) in enumerate(type_counts.items()):
                cols[i].metric(label=ptype, value=count)

            # Approximate Punch Frequency
            if 'frame_end' in df.columns:
                duration_frames = df['frame_end'].max() - df['frame_start'].min()
                fps = 30  # adjust this to your actual FPS
                duration_sec = duration_frames / fps if fps else 1
                punch_speed = len(df) / duration_sec if duration_sec > 0 else 0
                st.subheader("⚡ Speed Approximation")
                st.metric("Punches per Second", f"{punch_speed:.2f}")
            else:
                st.warning("Frame timing info missing — can't compute speed.")

            # Time-bucketed Frequency Chart
            if 'frame_start' in df.columns:
                df['time_sec'] = df['frame_start'] // 30  # adjust for your FPS
                time_counts = df.groupby('time_sec')['punch'].count()
                st.subheader("📈 Punch Frequency Over Time")
                fig1, ax1 = plt.subplots()
                time_counts.plot(kind='line', marker='o', ax=ax1)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Punches")
                ax1.set_title("Punches Per Second")
                st.pyplot(fig1)

            # Bar Chart of Punch Types
            st.subheader("📊 Punch Type Distribution")
            fig2, ax2 = plt.subplots()
            ax2.bar(type_counts.keys(), type_counts.values(), color='skyblue')
            #sns.barplot(x=list(type_counts.keys()), y=list(type_counts.values()), ax=ax2)
            ax2.set_ylabel("Count")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            st.pyplot(fig2)

            # Pie Chart of Punch Types
            st.subheader("🥧 Punch Share - Pie Chart")
            fig3, ax3 = plt.subplots()
            ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
            ax3.axis('equal')
            st.pyplot(fig3)
        else:
            st.info("🔍 No punch data found. Upload and process a video to see metrics.")

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
matplotlib
'''

with open("requirements.txt", "w") as f:
    f.write(requirements)
print("✅ requirements.txt saved")
