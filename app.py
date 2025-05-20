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
import numpy as np

# Global state: person_id -> punch state tracker
person_states = {}

# Tunable thresholds
VELOCITY_THRESHOLD = 15  # adjust based on pixel movement per frame
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

def classify_punch(keypoints, frame_idx):
    global person_states
    results = []

    for person_id, kpts in enumerate(keypoints):
        state = person_states.get(person_id, {
            "prev_kpts": kpts,
            "in_motion": {"left": False, "right": False},
            "frame_start": {"left": None, "right": None},
        })

        for side in ["left", "right"]:
            wrist = kpts.get(f"{side}_wrist")
            elbow = kpts.get(f"{side}_elbow")
            shoulder = kpts.get(f"{side}_shoulder")

            prev_wrist = state["prev_kpts"].get(f"{side}_wrist", wrist)

            if wrist is None or elbow is None or shoulder is None:
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
                punch_type = ""

                if elbow_angle < HOOK_ANGLE_THRESHOLD:
                    punch_type = "Hook"
                else:
                    if side == "left":
                        punch_type = "Jab"
                    else:
                        punch_type = "Cross"

                results.append({
                    "label": f"{side.capitalize()} {punch_type}",
                    "frame_start": frame_start,
                    "frame_end": frame_end
                })

                state["in_motion"][side] = False
                state["frame_start"][side] = None

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

            punches = classify_punch(keypoints,frame_idx)
            postures = check_posture(keypoints)
            gloves = detect_gloves(keypoints)

            annotated = draw_annotations(frame.copy(), keypoints, punches, postures, gloves)
            out_writer.write(annotated)

            for i in range(len(punches)):
                punch_log.append({
                    "video": uploaded_file.name,
                    "frame": frame_idx,
                    "person": i,
                    "punch": punches["label"],
                    "frame_start": punches["frame_start"],
                    "frame_end": punches["frame_end"],
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


        # Calculate speed (approx) if not present
        if 'speed (approx)' not in df_log.columns and 'frame_start' in df_log.columns and 'frame_end' in df_log.columns:
            df_log["duration_frames"] = df_log["frame_end"] - df_log["frame_start"]
            df_log["duration_frames"] = df_log["duration_frames"].replace(0, 1)
            df_log["speed (approx)"] = 30 / df_log["duration_frames"]  # Assuming 30 FPS
        #st.metric("Average Speed", f"{df_log['speed (approx)'].mean():.2f} punches/sec")

        if 'frame_start' not in df_log.columns or 'frame_end' not in df_log.columns:
          st.warning("Missing 'frame_start' or 'frame_end' in punch log. Speed cannot be calculated.")


        # ---- Performance Metrics ----
        st.header("📈 Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Punches", len(df_log))
        with col2:
            st.metric("Average Speed", f"{df_log['speed (approx)'].mean():.2f} punches/sec")
        with col3:
            most_common = df_log["punch_type"].value_counts().idxmax()
            st.metric("Most Frequent Punch", most_common)
        with col4:
            st.metric("Dummy Accuracy", "95%")  # Placeholder

        # ---- Punch Count Table ----
        st.subheader("🔢 Punch Type Count")
        counts = df_log["punch_type"].value_counts().reset_index()
        counts.columns = ["Punch Type", "Count"]
        st.dataframe(counts)

        # ---- Charts ----
        st.subheader("📊 Visual Analysis")

        tab1, tab2, tab3 = st.tabs(["Bar Chart", "Pie Chart", "Line Chart"])

        with tab1:
            st.write("### Punch Type Distribution")
            fig1, ax1 = plt.subplots()
            sns.barplot(x="Count", y="Punch Type", data=counts, ax=ax1, palette="viridis")
            st.pyplot(fig1)

        with tab2:
            st.write("### Punch Frequency Share")
            fig2, ax2 = plt.subplots()
            ax2.pie(counts["Count"], labels=counts["Punch Type"], autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
            ax2.axis("equal")
            st.pyplot(fig2)

        with tab3:
            st.write("### Punches Over Time")
            time_df = df_log.groupby(df_log["timestamp"].dt.floor("10s")).size().reset_index(name="Punch Count")
            fig3, ax3 = plt.subplots()
            ax3.plot(time_df["timestamp"], time_df["Punch Count"], marker="o")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Punch Count")
            st.pyplot(fig3)

        # ---- Download CSV ----
        st.subheader("📥 Download Log")
        csv_buffer = io.StringIO()
        df_log.to_csv(csv_buffer, index=False)
        st.download_button("Download CSV", data=csv_buffer.getvalue(), file_name="df_log.csv", mime="text/csv")

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
