import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

st.set_page_config(page_title="🥊 Boxing Analyzer", layout="wide")
st.title("🥊 Boxing Punch Detection (Red vs Blue Jerseys)")

# ------------------ Load MoveNet MultiPose ------------------
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

movenet = load_model()
# ------------------ Pose Detection ------------------

def detect_pose(image):
    input_img = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
    outputs = movenet.signatures["serving_default"](input_img)
    keypoints_all = outputs["output_0"].numpy()  # shape: (1, 6, 56)
    if keypoints_all.shape[-1] < 51:
        return []
    keypoints = keypoints_all[0, :, :51].reshape((6, 17, 3))
    return keypoints



# ------------------ Detect Jersey Color ------------------
def get_jersey_color(frame, keypoints):
    h, w, _ = frame.shape
    x_coords = keypoints[:, 1]
    y_coords = keypoints[:, 0]
    x_min = int(np.min(x_coords) * w)
    x_max = int(np.max(x_coords) * w)
    y_min = int(np.min(y_coords) * h)
    y_max = int(np.max(y_coords) * h)
    cropped = frame[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return "unknown"
    b, g, r = np.mean(cropped, axis=(0, 1))
    if r > 1.2 * b:
        return "red"
    elif b > 1.2 * r:
        return "blue"
    return "unknown"

# ------------------ Punch Detection ------------------
def is_punching(kp):
    lw = kp[9]     # left wrist
    rw = kp[10]    # right wrist
    ls = kp[5]     # left shoulder
    rs = kp[6]     # right shoulder

    lw_punch = abs(lw[1] - ls[1]) > 0.15 and lw[2] > 0.3
    rw_punch = abs(rw[1] - rs[1]) > 0.15 and rw[2] > 0.3

    if lw_punch:
        return "Left Punch"
    elif rw_punch:
        return "Right Punch"
    return None

# ------------------ Process Video ------------------
def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints_list = detect_pose(frame)

        for kp in keypoints_list:
            if kp[0][2] < 0.3:
                continue

            jersey = get_jersey_color(frame, kp)
            punch = is_punching(kp)

            for (y, x, c) in kp:
                if c > 0.3:
                    cx, cy = int(x * width), int(y * height)
                    color = (0, 0, 255) if jersey == "red" else (255, 0, 0)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

            label = f"{jersey.upper()}"
            if punch:
                label += f" - {punch}"
            x_nose = int(kp[0][1] * width)
            y_nose = int(kp[0][0] * height)
            cv2.putText(frame, label, (x_nose, y_nose - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    return out_path

# ------------------ Streamlit UI ------------------
uploaded_file = st.file_uploader("📤 Upload a boxing video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with st.spinner("Processing video..."):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        temp_file.close()
        video_path = temp_file.name

        output_path = process_video(video_path)

        st.success("✅ Video processing complete!")
        st.subheader("🎥 Annotated Video Output")
        st.video(output_path)
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            label="📥 Download Annotated Video",
            data=video_bytes,
            file_name="annotated_boxing_video.mp4",
            mime="video/mp4"
        )




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
