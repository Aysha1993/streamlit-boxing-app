import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from sklearn import svm
from joblib import dump

# Load MoveNet MultiPose model
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

# Draw keypoints and annotations
def draw_annotations(frame, keypoints, punches, postures, gloves):
    for idx, person in enumerate(keypoints):
        if len(person) != 17:
            continue

        for i, kp in enumerate(person):
            if kp[2] > 0.2:
                x, y = int(kp[1] * frame.shape[1]), int(kp[0] * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Position to draw text
        base_x, base_y = int(person[0][1] * frame.shape[1]), int(person[0][0] * frame.shape[0])
        text_y = base_y - 20 if base_y - 20 > 20 else base_y + 20

        label = f"{punches[idx]}, {postures[idx]}, {gloves[idx]}"
        cv2.putText(frame, label, (base_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

# Extract keypoints
def extract_keypoints(results):
    keypoints = []
    for person in results['output_0'][0]:
        person_kps = []
        for i in range(17):
            y, x, score = person[i*3:(i+1)*3]
            person_kps.append([y, x, score])
        keypoints.append(person_kps)
    return keypoints

# Classify punch type with left/right
def classify_punch(keypoints):
    punch_type = []
    for person in keypoints:
        if len(person) != 17:
            punch_type.append("unknown")
            continue
        lwrist, rwrist = person[9], person[10]
        lshoulder, rshoulder = person[5], person[6]
        lelbow, relbow = person[7], person[8]

        if lwrist[2] > 0.2 and lshoulder[2] > 0.2 and lwrist[0] < lshoulder[0]:
            punch_type.append("Left Jab")
        elif rwrist[2] > 0.2 and rshoulder[2] > 0.2 and rwrist[0] < rshoulder[0]:
            punch_type.append("Right Cross")
        elif lelbow[2] > 0.2 and abs(lelbow[1] - lwrist[1]) > 0.1:
            punch_type.append("Left Hook")
        elif relbow[2] > 0.2 and abs(relbow[1] - rwrist[1]) > 0.1:
            punch_type.append("Right Hook")
        else:
            punch_type.append("Guard")
    return punch_type

# Posture checking for multiple joints
def check_posture(keypoints):
    posture_feedback = []
    for person in keypoints:
        if len(person) != 17:
            posture_feedback.append("unknown")
            continue
        msg = []
        # Elbow drop
        lelbow, relbow = person[7], person[8]
        lhip, rhip = person[11], person[12]
        if lelbow[0] > lhip[0]:
            msg.append("Left Elbow ↓")
        if relbow[0] > rhip[0]:
            msg.append("Right Elbow ↓")
        # Shoulder drop
        if person[5][0] > person[11][0]:
            msg.append("Left Shoulder ↓")
        if person[6][0] > person[12][0]:
            msg.append("Right Shoulder ↓")
        # Knees
        if person[13][2] > 0.2 and person[15][2] > 0.2:
            if person[15][0] < person[13][0] - 0.05:
                msg.append("Left Knee Bent")
        if person[14][2] > 0.2 and person[16][2] > 0.2:
            if person[16][0] < person[14][0] - 0.05:
                msg.append("Right Knee Bent")
        # Wrists height
        if person[9][0] > person[7][0]:
            msg.append("Left Wrist ↓")
        if person[10][0] > person[8][0]:
            msg.append("Right Wrist ↓")
        posture_feedback.append(", ".join(msg) if msg else "Good Posture")
    return posture_feedback

# Glove detection
def detect_gloves(keypoints):
    gloves = []
    for person in keypoints:
        lwrist, rwrist = person[9], person[10]
        gloves.append(f"Gloves: L-{'yes' if lwrist[2] > 0.2 else 'no'} R-{'yes' if rwrist[2] > 0.2 else 'no'}")
    return gloves

# Streamlit UI
st.title("🥊 Boxing Analyzer with Annotations")

uploaded_files = st.file_uploader("Upload MP4 boxing videos", type=["mp4"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_file.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        base_name = os.path.splitext(uploaded_file.name)[0]
        out_path = f"/tmp/{base_name}_out.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        punch_log = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (256, 256))
            img = tf.convert_to_tensor(resized, dtype=tf.uint8)
            input_tensor = tf.expand_dims(img, axis=0)
            input_tensor = tf.cast(input_tensor, dtype=tf.int32)

            results = model.signatures['serving_default'](input_tensor)
            keypoints = extract_keypoints(results)
            punches = classify_punch(keypoints)
            postures = check_posture(keypoints)
            gloves = detect_gloves(keypoints)

            frame = draw_annotations(frame, keypoints, punches, postures, gloves)

            for i, punch in enumerate(punches):
                punch_log.append({
                    "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "person": i,
                    "punch": punches[i],
                    "posture": postures[i],
                    "gloves": gloves[i]
                })

            out.write(frame)

        cap.release()
        out.release()

        st.video(out_path)
        st.success("✅ Video processed and annotated!")
        #csv file
        df = pd.DataFrame(punch_log)
        st.write(f"Total punch entries: {len(punch_log)}")
        st.dataframe(df)
        
        if not df.empty:
            st.dataframe(df.head(10))
        
            # Use StringIO for download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{base_name}_log.csv",
                mime="text/csv"
            )
        else:
            st.warning("No punch data was extracted.")


        #csv_dest = f"/tmp/{base_name}_punch_log.csv"
        #df.to_csv(csv_dest, index=False)
        #st.download_button("Download CSV", csv_dest, file_name=f"{base_name}_log.csv")

        model_dest = f"/tmp/{base_name}_svm_model.joblib"
        if st.button(f"Train SVM on {uploaded_file.name}"):
            if 'punch' in df.columns:
                X = df[['frame', 'person']]
                y = df['punch']
                clf = svm.SVC()
                clf.fit(X, y)
                dump(clf, model_dest)
                st.success("SVM trained and saved ✅")
