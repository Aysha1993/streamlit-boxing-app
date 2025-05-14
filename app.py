import streamlit as st
import tensorflow\_hub as hub
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

# Load MoveNet Multipose model

@st.cache\_resource
def load\_model():
model = hub.load("[https://tfhub.dev/google/movenet/multipose/lightning/1](https://tfhub.dev/google/movenet/multipose/lightning/1)")
return model.signatures\['serving\_default']

# Detect poses from frame

def detect\_poses(frame, model):
input\_size = 256
img = tf.image.resize\_with\_pad(tf.expand\_dims(frame, axis=0), input\_size, input\_size)
input\_img = tf.cast(img, dtype=tf.int32)
outputs = model(input\_img)
keypoints\_with\_scores = outputs\['output\_0'].numpy()\[:, :, :51].reshape((6, 17, 3))

```
keypoints = []
for person in keypoints_with_scores:
    if np.mean(person[:, 2]) > 0.2:
        keypoints.append(person.tolist())
return keypoints
```

# Filter top 2 confident persons (assumed to be boxers)

def filter\_top\_two\_persons(keypoints):
scored = \[]
for idx, kp in enumerate(keypoints):
score = np.mean(\[s for (\_, *, s) in kp])
scored.append((score, idx))
top\_two = sorted(scored, reverse=True)\[:2]
return \[keypoints\[i] for (*, i) in top\_two]

# Draw skeleton on frame

def draw\_skeleton(frame, keypoints):
height, width, \_ = frame.shape
keypoint\_edges = \[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(9,10),
(11,12),(11,13),(13,15),(12,14),(14,16)]
for person in keypoints:
for edge in keypoint\_edges:
p1 = person\[edge\[0]]
p2 = person\[edge\[1]]
if p1\[2] > 0.2 and p2\[2] > 0.2:
x1, y1 = int(p1\[1]\*width), int(p1\[0]\*height)
x2, y2 = int(p2\[1]\*width), int(p2\[0]\*height)
cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
for idx, kp in enumerate(person):
if kp\[2] > 0.2:
x, y = int(kp\[1]\*width), int(kp\[0]\*height)
cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
return frame

# Detect gloves from wrist keypoints

def detect\_gloves(keypoints):
gloves = \[]
for person in keypoints:
left\_wrist = person\[9]
right\_wrist = person\[10]
if left\_wrist\[2] > 0.3:
gloves.append(('Left Glove', left\_wrist))
if right\_wrist\[2] > 0.3:
gloves.append(('Right Glove', right\_wrist))
return gloves

# Annotate detections

def annotate(frame, gloves):
height, width, \_ = frame.shape
for name, (y, x, c) in gloves:
cx, cy = int(x \* width), int(y \* height)
cv2.putText(frame, name, (cx, cy - 10), cv2.FONT\_HERSHEY\_SIMPLEX, 0.5, (255, 0, 0), 2)
cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
return frame

# Process and annotate video

def process\_video(input\_path, model):
cap = cv2.VideoCapture(input\_path)
width = int(cap.get(cv2.CAP\_PROP\_FRAME\_WIDTH))
height = int(cap.get(cv2.CAP\_PROP\_FRAME\_HEIGHT))
fps = int(cap.get(cv2.CAP\_PROP\_FPS))
out\_path = tempfile.mktemp(suffix='.mp4')
fourcc = cv2.VideoWriter\_fourcc(\*'mp4v')
out = cv2.VideoWriter(out\_path, fourcc, fps, (width, height))

```
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raw_keypoints = detect_poses(frame_rgb, model)
    keypoints = filter_top_two_persons(raw_keypoints)  # Only top 2 confident persons assumed to be players
    gloves = detect_gloves(keypoints)
    frame = draw_skeleton(frame, keypoints)
    frame = annotate(frame, gloves)
    out.write(frame)

cap.release()
out.release()
return out_path
```

# Streamlit UI

st.title("Boxing Pose Estimator with Glove Detection")
model = load\_model()
video\_file = st.file\_uploader("Upload Boxing Video", type=\["mp4", "mov"])

if video\_file:
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(video\_file.read())
st.video(tfile.name)
with st.spinner("Processing video..."):
annotated\_path = process\_video(tfile.name, model)
st.success("Video processed!")
st.video(annotated\_path)
with open(annotated\_path, "rb") as f:
st.download\_button("Download Annotated Video", f, file\_name="annotated\_output.mp4")

```
    df = pd.DataFrame(punch_log)
    st.dataframe(df)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download CSV", csv_buffer.getvalue(), file_name=f"{uploaded_file.name}_log.csv", mime="text/csv")

    base_name = os.path.splitext(uploaded_file.name)[0]
    model_dest = f"/tmp/{base_name}_svm_model.joblib"

    if st.button(f"Train SVM on {uploaded_file.name}"):
        if 'punch' in df.columns:
            X = df[['frame', 'person']]
            y = df['punch']
            clf = svm.SVC()
            clf.fit(X, y)
            dump(clf, model_dest)
            st.success("SVM trained and saved ✅")
