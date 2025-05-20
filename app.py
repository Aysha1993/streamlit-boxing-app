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

# File uploader
uploaded_files = st.file_uploader("Upload multiple boxing videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)

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

            resized = cv2.resize(frame, (256, 256))
            input_tensor = tf.convert_to_tensor(resized[None, ...], dtype=tf.int32)
            results = model.signatures['serving_default'](input_tensor)
            keypoints = extract_keypoints(results)

            if not keypoints:
                out_writer.write(frame)
                continue

            punches = classify_punch(keypoints)
            postures = check_posture(keypoints)
            gloves = detect_gloves(keypoints)

            annotated = draw_annotations(frame.copy(), keypoints, punches, postures, gloves)
            out_writer.write(annotated)

            for i in range(len(punches)):
                punch_log.append({
                    "video": uploaded_file.name,
                    "frame": frame_idx,
                    "person": i,
                    "punch": punches[i],
                    "posture": postures[i],
                    "gloves": gloves[i],
                    "keypoints": keypoints[i]
                })

            frame_idx += 1
            if frame_idx % 5 == 0:              
              total_progress = (idx + frame_idx / total_frames) / len(uploaded_files)
              progress_bar.progress(min(total_progress, 1.0))

        cap.release()
        out_writer.release()

        # FFmpeg encode
        final_output = os.path.join(temp_dir, f"final_{uploaded_file.name}")
        ffmpeg.input(raw_output).output(final_output, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True)

        st.video(final_output)
        with open(final_output, "rb") as f:
            st.download_button("📥 Download Annotated Video", f, file_name=f"annotated_{uploaded_file.name}", mime="video/mp4")

        df = pd.DataFrame(punch_log)
        if df.empty:
            st.warning("⚠️ No punch data found.")
            continue

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
matplotlib
'''

with open("requirements.txt", "w") as f:
    f.write(requirements)
print("✅ requirements.txt saved")
