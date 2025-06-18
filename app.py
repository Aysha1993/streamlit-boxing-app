import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib
import ffmpeg


# Load MoveNet SinglePose or MultiPose model
@st.cache_resource
def load_movenet_model():
    model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"  # or multipose/lightning
    model = hub.load(model_url)
    return model.signatures['serving_default']

# Load classifier (e.g., from sklearn)
@st.cache_resource
def load_classifier(model_path="punch_classifier.pkl"):
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf
def preprocess_keypoints(keypoints):
    # keypoints shape: [1, 1, 17, 3]
    keypoints = keypoints[0, 0, :, :3]  # x, y, score
    flattened = keypoints.flatten()
    return flattened  # shape: (51,)

# ---- Save video using OpenCV ----
def save_video(frames, fps, width, height, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        if frame.dtype != 'uint8':
            frame = frame.astype('uint8')
        out.write(frame)
    out.release()

# ---- Re-encode with FFmpeg for browser compatibility ----
def reencode_with_ffmpeg(input_path, output_path):
    ffmpeg.input(input_path).output(output_path, vcodec='libx264', acodec='aac', strict='experimental', pix_fmt='yuv420p').run(overwrite_output=True)

def draw_skeleton(frame, keypoints, label=None):
    keypoints = keypoints[0, 0, :, :2]
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    if label:
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    return frame

def extract_and_predict(video_path, model, clf):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    
    output_frames = []
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
        input_img = tf.cast(img, dtype=tf.int32)
        keypoints = model(input_img)  # SinglePose: [1, 1, 17, 3]

        keypoint_data = preprocess_keypoints(keypoints['output_0'].numpy())
        # st.write(f"Classifier loaded: {type(clf)}")
        label = clf.predict([keypoint_data])[0]
        predictions.append(label)

        annotated = draw_skeleton(frame.copy(), keypoints['output_0'].numpy(), label)
        output_frames.append(annotated)

    cap.release()
    return output_frames, predictions, fps, width, height



# ------------------- Streamlit GUI -------------------

st.title("🥊 Punch Detection using MoveNet + Classifier")

# clf = joblib.load("punch_classifier_model.joblib")
uploaded_model = st.file_uploader("Upload Trained Classifier (.joblib)", type=["joblib"])
clf = None  # Initialize

if uploaded_model is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_model:
            tmp_model.write(uploaded_model.read())
            tmp_model.flush()
            clf = joblib.load(tmp_model.name)

        # st.success(f"✅ Model loaded: {type(clf)}")
    except Exception as e:
        st.error(f"❌ Failed to load classifier: {e}")
        clf = None


uploaded_file = st.file_uploader("Upload Boxing Video", type=["mp4", "avi", "mov"])
model = load_movenet_model()
if uploaded_file and clf:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.info("⏳ Processing video and predicting punches...")
    frames, preds, fps, width, height = extract_and_predict(video_path, model, clf)

    raw_output_path = os.path.join(tempfile.gettempdir(), "raw_output.mp4")
    final_output_path = os.path.join(tempfile.gettempdir(), "predicted_output.mp4")

    save_video(frames, fps, width, height, raw_output_path)
    reencode_with_ffmpeg(raw_output_path, final_output_path)

    st.success("✅ Prediction complete! Showing result:")

    with open(final_output_path, 'rb') as f:
        st.video(f.read())

    # Save prediction log
    csv_path = os.path.join(tempfile.gettempdir(), "punch_predictions.csv")
    df = pd.DataFrame({'frame': list(range(len(preds))), 'punch_type': preds})
    df.to_csv(csv_path, index=False)

    st.download_button("Download Prediction CSV", data=open(csv_path, "rb"), file_name="punch_predictions.csv")

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
