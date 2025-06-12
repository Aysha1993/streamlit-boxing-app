import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import tempfile
import os
import base64

# --- Load MoveNet MultiPose Model ---
movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1").signatures['serving_default']

def detect_persons(frame):
    input_img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    input_img = tf.cast(input_img, dtype=tf.int32)
    outputs = movenet(input_img)
    keypoints_with_scores = outputs['output_0'].numpy()  # shape: (1, 6, 56)

    persons = []
    for person in keypoints_with_scores[0]:
        if person[55] < 0.2:
            continue
        keypoints = person[:51].reshape((17, 3))
        bbox = person[51:55]  # [ymin, xmin, ymax, xmax]
        kps = np.array([[kp[1], kp[0]] for kp in keypoints])  # (x, y)
        scores = keypoints[:, 2]
        persons.append({
            'keypoints': kps,
            'scores': scores,
            'bbox_norm': [bbox[1], bbox[0], bbox[3], bbox[2]]  # [xmin, ymin, xmax, ymax]
        })
    return persons

def draw_ids(frame, tracked):
    h, w, _ = frame.shape
    for tid, person in tracked:
        x1n, y1n, x2n, y2n = person['bbox_norm']
        x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for x, y in person['keypoints']:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
    return frame

def process_video(video_path, output_path="output_sort.mp4"):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracked_positions = {}  # {temp_id: [(cx, cy)]}
    movement_scores = {}    # {temp_id: float}
    id_counter = 1
    frame_num = 0
    id_map = {}             # temporary id to assigned boxer ID (1 or 2)
    boxer_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        persons = detect_persons(frame)
        person_with_boxes = []
        for person in persons:
            x1n, y1n, x2n, y2n = person['bbox_norm']
            x1, y1, x2, y2 = int(x1n * width), int(y1n * height), int(x2n * width), int(y2n * height)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            person_with_boxes.append((person, [x1, y1, x2, y2], (cx, cy)))

        # Assign temp IDs
        new_tracked = []
        for person, abs_box, center in person_with_boxes:
            matched_temp_id = None
            for tid, positions in tracked_positions.items():
                prev_cx, prev_cy = positions[-1]
                dist = np.linalg.norm(np.array(center) - np.array([prev_cx, prev_cy]))
                if dist < 80:
                    matched_temp_id = tid
                    break

            if matched_temp_id is None:
                matched_temp_id = id_counter
                id_counter += 1

            # Update movement history
            if matched_temp_id not in tracked_positions:
                tracked_positions[matched_temp_id] = [center]
                movement_scores[matched_temp_id] = 0
            else:
                prev_cx, prev_cy = tracked_positions[matched_temp_id][-1]
                dx = center[0] - prev_cx
                dy = center[1] - prev_cy
                movement_scores[matched_temp_id] += np.sqrt(dx**2 + dy**2)
                tracked_positions[matched_temp_id].append(center)

            new_tracked.append((matched_temp_id, person))

        # After 100 frames, choose 2 most active IDs as boxers
        if frame_num == 100 and len(movement_scores) >= 2:
            top_ids = sorted(movement_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            boxer_ids = set([tid for tid, _ in top_ids])
            id_map = {tid: bid + 1 for bid, tid in enumerate(boxer_ids)}

        # Final tracked list with boxer IDs only
        final_tracked = []
        for tid, person in new_tracked:
            if tid in boxer_ids:
                final_tracked.append((id_map[tid], person))

        annotated = draw_ids(frame, final_tracked)
        out.write(annotated)

    cap.release()
    out.release()
    st.info(f"✅ Constant ID tracking (boxers only) saved: {output_path}")

def play_video(video_path):
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        b64_encoded = base64.b64encode(video_bytes).decode()
        video_html = f'''
            <video width="700" controls>
                <source src="data:video/mp4;base64,{b64_encoded}" type="video/mp4">
            </video>
        '''
        st.markdown(video_html, unsafe_allow_html=True)

# --- Streamlit UI ---
st.title("🥊 Boxing Analyzer - Constant ID Tracking (Boxers Only)")

uploaded_files = st.file_uploader("Upload boxing video", type=["mp4", "avi", "mov"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_video_path, 'wb') as f:
            f.write(uploaded_file.read())

        output_path = os.path.join(temp_dir, "output_sort.mp4")
        process_video(temp_video_path, output_path)

        st.success("✅ Processed video with constant ID tracking.")
        play_video(output_path)

        with open(output_path, "rb") as file:
            st.download_button("📥 Download Tracked Video", file, "tracked_output.mp4", "video/mp4")

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
