import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from datetime import datetime

# ------------------ SORT Tracker (Simplified) ------------------
class Sort:
    def __init__(self):
        self.counter = 0

    def update(self, detections):
        results = []
        for det in detections:
            self.counter += 1
            x, y, w, h = det
            results.append([x, y, w, h, self.counter % 2])  # Fake ID 0 or 1
        return results

# ------------------ MoveNet Utilities ------------------
movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
model = movenet.signatures['serving_default']

def detect_keypoints(frame):
    image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    input_img = tf.cast(image, dtype=tf.int32)
    outputs = model(input_img)
    keypoints = outputs["output_0"].numpy()[0]

    input_h, input_w = frame.shape[:2]
    scale_y = input_h / 256
    scale_x = input_w / 256
    return keypoints, scale_x, scale_y

def get_persons_from_output(output, threshold=0.2):
    persons = []
    for person in output:
        keypoints = person[:51].reshape((17, 3))
        if keypoints[0, 2] < threshold:
            continue
        persons.append({"keypoints": keypoints})
    return persons

def get_bbox_from_keypoints(keypoints):
    valid = keypoints[keypoints[:, 2] > 0.2][:, :2]
    if len(valid) == 0:
        return None
    x1, y1 = valid.min(axis=0)
    x2, y2 = valid.max(axis=0)
    w, h = x2 - x1, y2 - y1
    if w < 1 or h < 1:
        return None
    return [x1, y1, w, h]

# ------------------ Streamlit App ------------------
st.set_page_config(layout="wide")
st.title("🎣 MoveNet MultiPose + SORT Boxer Tracker")

uploaded_file = st.file_uploader("Upload a boxing video", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    tracker = Sort()
    boxer_ids = set()

    # Get video properties
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define output video
    output_path = os.path.join(tempfile.gettempdir(), f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints_with_scores, scale_x, scale_y = detect_keypoints(input_frame)
        persons = get_persons_from_output(keypoints_with_scores)

        boxes = []
        for person in persons:
            # Scale keypoints to original frame size
            person["keypoints"][:, 0] *= scale_x
            person["keypoints"][:, 1] *= scale_y

            bbox = get_bbox_from_keypoints(person["keypoints"])
            if bbox:
                boxes.append(bbox)

        if len(boxes) == 0:
            out.write(frame)
            stframe.image(frame, channels="BGR")
            continue

        track_results = tracker.update(np.array(boxes))

        for person, track in zip(persons, track_results):
            x, y, w, h, track_id = track
            track_id = int(track_id)
            person["id"] = track_id

        if frame_count < 30:
            for person in persons:
                boxer_ids.add(person["id"])
            boxer_ids = set(list(boxer_ids)[:2])
        frame_count += 1

        for person in persons:
            if person["id"] not in boxer_ids:
                continue
            kps = person["keypoints"]

            # Draw skeleton keypoints
            for xk, yk, c in kps:
                if c > 0.2:
                    cv2.circle(frame, (int(xk), int(yk)), 3, (0, 255, 0), -1)

            # Draw bounding box
            bbox = get_bbox_from_keypoints(kps)
            if bbox:
                x, y, w, h = map(int, bbox)
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)


            # Draw label
            cv2.putText(frame, f"Boxer {person['id']}", (int(kps[0][0]), int(kps[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            st.info(f"boxes={boxes} ,bbox={bbox},frame_count ={frame_count}")

        out.write(frame)
        stframe.image(frame, channels="BGR")

    cap.release()
    out.release()
    st.success("🚀 Video Processing Complete!")

    with open(output_path, "rb") as f:
        st.download_button("📥 Download Annotated Video", f, file_name="annotated_output.mp4")

# --- requirements.txt generator ---
requirements = '''streamlit
tensorflow
tensorflow_hub
opencv-python-headless
filterpy
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
