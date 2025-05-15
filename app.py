import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
import ffmpeg
from sklearn import svm
from joblib import dump
import io
from sklearn.model_selection import train_test_split

# Streamlit setup
st.set_option('client.showErrorDetails', True)
st.title("🥊 Boxing Analyzer with Punches, Posture & Gloves")

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

# Upload and process videos
uploaded_files = st.file_uploader("Upload multiple boxing videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)

if uploaded_files:
    all_logs = []  # Collect all logs here

    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")
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

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            for i in range(len(punches)):
                punch_log.append({
                    "video": uploaded_file.name,
                    "frame": frame_id,
                    "person": i,
                    "punch": punches[i],
                    "posture": postures[i],
                    "gloves": gloves[i],
                    "keypoints": keypoints[i]
                })

        cap.release()
        out_writer.release()

        final_output = os.path.join(temp_dir, f"final_{uploaded_file.name}")
        ffmpeg.input(raw_output).output(final_output, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True)

        st.video(final_output)
        st.success(f"✅ Annotated video ready for {uploaded_file.name}")

        with open(final_output, "rb") as f:
            st.download_button("📥 Download Annotated Video", f, file_name=f"annotated_{uploaded_file.name}", mime="video/mp4")     

        df = pd.DataFrame(punch_log)
        
        def expand_keypoints(keypoints):
            #  Convert stringified list to actual list if needed
            if isinstance(keypoints, str):
                keypoints = json.loads(keypoints)
        
            #  Sanity check
            if not isinstance(keypoints, list):
                return pd.Series()  # skip if not list
        
            try:      
                x_coords = [kp['x'] for kp in keypoints]
                y_coords = [kp['y'] for kp in keypoints]
                scores   = [kp['score'] for kp in keypoints]         
                data = {
                    f'x_{i}': x for i, x in enumerate(x_coords)
                } | {
                    f'y_{i}': y for i, y in enumerate(y_coords)
                } | {
                    f's_{i}': s for i, s in enumerate(scores)
                }         
                return pd.Series(data)    
            except (KeyError, TypeError):
                return pd.Series()  # return empty if malformed    
        # Apply to expand each keypoint list to individual columns
        keypoints_df = df['keypoints'].apply(expand_keypoints)
        df_expanded = pd.concat([df.drop(columns=['keypoints']), keypoints_df], axis=1)
        
        # Show table in Streamlit and allow download
        st.dataframe(df_expanded)
        
        csv_buffer = io.StringIO()
        df_expanded.to_csv(csv_buffer, index=False)
        st.download_button("📥 Download CSV", csv_buffer.getvalue(), file_name=f"{uploaded_file.name}_log.csv", mime="text/csv")

        all_logs.extend(punch_log)

        # Optional: SVM training per file
        base_name = os.path.splitext(uploaded_file.name)[0]
        model_dest = f"/tmp/{base_name}_svm_model.joblib"

        if st.button(f"Train SVM on {uploaded_file.name}"):
            if 'punch' in df.columns:
                # Example: Simple SVM using keypoints               
                if all_logs:
                    df_all = pd.DataFrame(all_logs)
                    expanded = df_all['keypoints'].apply(expand_keypoints)
                    df_full = pd.concat([df_all.drop(columns=['keypoints']), expanded], axis=1)
                
                    X = df_full[[col for col in df_full.columns if 'kp_' in col]]
                    y = df_full['punch']
                
                    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
                    clf = svm.SVC()
                    clf.fit(X_train, y_train)
                
                    acc = clf.score(X_test, y_test)
                    st.info(f"Punch Classifier Accuracy: {acc:.2f}")
                
                    dump(clf, "punch_svm_model.joblib")
                    st.download_button("📥 Download SVM Model", data=open("punch_svm_model.joblib", "rb"), file_name="svm_model.joblib")
                    st.success(f"✅ SVM trained and saved for {uploaded_file.name}")

    # Save consolidated CSV after all videos
    if all_logs:
        st.subheader("📦 All Video Logs Summary")
        all_df = pd.DataFrame(all_logs)
        st.dataframe(all_df)

        full_csv = io.StringIO()
        all_df.to_csv(full_csv, index=False)
        st.download_button("📥 Download Combined CSV for All Videos", full_csv.getvalue(), file_name="combined_video_logs.csv", mime="text/csv")
