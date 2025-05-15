import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
import shutil
import ffmpeg
from sklearn import svm
from joblib import dump
import io



# Load MoveNet MultiPose model
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']

# Define keypoint names
KEYPOINT_DICT = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

# Draw skeleton connections
KEYPOINT_EDGE_INDS = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10),
                      (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

def detect_keypoints(img):
    img_rgb = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    input_img = tf.cast(img_rgb, dtype=tf.int32)
    result = movenet(input_img)
    keypoints_with_scores = result['output_0'].numpy()
    return keypoints_with_scores[0]

def extract_valid_keypoints(keypoints_with_scores, threshold=0.2):
    valid_people = []
    for person in keypoints_with_scores:
        if person[55] < threshold:  # detection confidence
            continue
        keypoints = person[:51].reshape((17, 3))
        valid_people.append(keypoints)
    return valid_people

def detect_gloves(keypoints, distance_thresh=0.03):
    gloves = []
    for kp in keypoints:
        lw, le = kp[9], kp[7]
        rw, re = kp[10], kp[8]

        def has_glove(wrist, elbow):
            if wrist[2] > 0.1 and elbow[2] > 0.1:
                dist = np.sqrt((wrist[1] - elbow[1])**2 + (wrist[0] - elbow[0])**2)
                return dist > distance_thresh
            return False

        left_glove = "yes" if has_glove(lw, le) else "no"
        right_glove = "yes" if has_glove(rw, re) else "no"
        gloves.append((left_glove, right_glove))
    return gloves

def draw_annotations(img, keypoints, punches, postures, gloves):
    for i, person in enumerate(keypoints):
        for edge in KEYPOINT_EDGE_INDS:
            p1, p2 = edge
            y1, x1, c1 = person[p1]
            y2, x2, c2 = person[p2]
            if c1 > 0.2 and c2 > 0.2:
                pt1 = (int(x1 * img.shape[1]), int(y1 * img.shape[0]))
                pt2 = (int(x2 * img.shape[1]), int(y2 * img.shape[0]))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        for idx, (y, x, c) in enumerate(person):
            if c > 0.2:
                center = (int(x * img.shape[1]), int(y * img.shape[0]))
                cv2.circle(img, center, 3, (0, 0, 255), -1)
        # Add punch, posture, glove text
        if punches and i < len(punches):
            cv2.putText(img, f"Punch: {punches[i]}", (10, 30 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        if postures and i < len(postures):
            cv2.putText(img, f"Posture: {postures[i]}", (10, 50 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if gloves and i < len(gloves):
            cv2.putText(img, f"Gloves: {gloves[i]}", (10, 70 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def classify_punch(keypoints):
    punches = []
    for kp in keypoints:
        lw, le, ls = kp[9], kp[7], kp[5]
        rw, re, rs = kp[10], kp[8], kp[6]
        left_punch = lw[0] < le[0] < ls[0] if lw[2] > 0.2 and le[2] > 0.2 and ls[2] > 0.2 else False
        right_punch = rw[0] < re[0] < rs[0] if rw[2] > 0.2 and re[2] > 0.2 and rs[2] > 0.2 else False
        if left_punch:
            punches.append("Left Jab")
        elif right_punch:
            punches.append("Right Jab")
        else:
            punches.append("No Punch")
    return punches

def check_posture(keypoints):
    posture_status = []
    for kp in keypoints:
        le, re = kp[7], kp[8]
        if le[2] > 0.2 and le[0] > 0.6:
            posture_status.append("Elbow Drop Left")
        elif re[2] > 0.2 and re[0] > 0.6:
            posture_status.append("Elbow Drop Right")
        else:
            posture_status.append("Good")
    return posture_status

# Run analyzer on video
def analyze_video(video_path, output_path="output_annotated.mp4"):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    punch_log = []

    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break
        keypoints_with_scores = detect_keypoints(frame)
        keypoints = extract_valid_keypoints(keypoints_with_scores)

        gloves_all = detect_gloves(keypoints)
        print(f"Detected {len(keypoints)} people, Gloves: {gloves_all}")

        keypoints_with_gloves = []
        gloves_filtered = []
        indices_with_gloves = []

        for i, (kp, (lg, rg)) in enumerate(zip(keypoints, gloves_all)):
            if lg == "yes" or rg == "yes":
                keypoints_with_gloves.append(kp)
                gloves_filtered.append((lg, rg))
                indices_with_gloves.append(i)

        # Fallback: if no gloves detected, use all
        if not keypoints_with_gloves:
            keypoints_with_gloves = keypoints
            gloves_filtered = gloves_all
            indices_with_gloves = list(range(len(keypoints)))

        punches = classify_punch(keypoints_with_gloves)
        postures = check_posture(keypoints_with_gloves)

        # Log punches
        for i, kp in enumerate(keypoints_with_gloves):
            punch_log.append({
                "Frame": cap.get(cv2.CAP_PROP_POS_FRAMES),
                "Person_ID": indices_with_gloves[i],
                "Punch": punches[i],
                "Posture": postures[i],
                "Gloves": gloves_filtered[i]
            })

        annotated = draw_annotations(frame, keypoints_with_gloves, punches, postures, gloves_filtered)
        out.write(annotated)

    cap.release()
    out.release()

    # Save punch log
    df = pd.DataFrame(punch_log)
    df.to_csv("punch_log.csv", index=False)
    print("Punch log saved to punch_log.csv")

    """base_name = os.path.splitext(uploaded_file.name)[0]
    model_dest = f"/tmp/{base_name}_svm_model.joblib"
    
    if st.button(f"Train SVM on {uploaded_file.name}"):
        if 'punch' in df.columns:
            X = df[['frame', 'person']]
            y = df['punch']
            clf = svm.SVC()
            clf.fit(X, y)
            dump(clf, model_dest)
            st.success("SVM trained and saved ✅")"""
