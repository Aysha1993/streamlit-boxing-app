import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os
import joblib
import ffmpeg

# 🧹 Clean old temp files
temp_dir = tempfile.gettempdir()
for f in ["movenet_punches.csv", "punch_comparison.csv", "model_predictions.csv", "raw_output.mp4", "predicted_output.mp4"]:
    try:
        os.remove(os.path.join(temp_dir, f))
    except FileNotFoundError:
        pass

# 🚀 Load MoveNet model
@st.cache_resource
def load_movenet_model():
    model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    model = hub.load(model_url)
    return model.signatures['serving_default']

# 📦 Preprocess keypoints
def preprocess_keypoints(keypoints):
    keypoints = keypoints[0, 0, :, :3]
    flattened = keypoints.flatten()
    return flattened

# 🤖 Rule-based punch logic
def rule_based_prediction(keypoints_flat):
    kp = np.array(keypoints_flat).reshape(17, 3)
    if kp[9][1] < kp[7][1]:
        return "Jab"
    elif kp[10][1] < kp[8][1]:
        return "Cross"
    return "none"

# 🧍 Draw pose
def draw_skeleton(frame, keypoints, label=None):
    keypoints = keypoints[0, 0, :, :2]
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    if label:
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    return frame

# 🎥 Save raw video
def save_video(frames, fps, width, height, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame.astype('uint8'))
    out.release()

# ✅ Deduplicate predictions
def deduplicate_punches(preds_rule):
    last_label = "none"
    clean_punches = []
    for label in preds_rule:
        if label != "none" and label != last_label:
            clean_punches.append(label)
        last_label = label
    return clean_punches

# 🔄 Re-encode video
def reencode_with_ffmpeg(input_path, output_path):
    ffmpeg.input(input_path).output(output_path, vcodec='libx264', acodec='aac', pix_fmt='yuv420p').run(overwrite_output=True)

# 🔍 Main prediction
def extract_and_predict(video_path, model, clf):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))

    output_frames = []
    model_preds = []
    rule_preds = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
        input_img = tf.cast(img, dtype=tf.int32)
        keypoints = model(input_img)

        keypoint_data = preprocess_keypoints(keypoints['output_0'].numpy())
        model_label = clf.predict([keypoint_data])[0]
        rule_label = rule_based_prediction(keypoint_data)

        model_preds.append(model_label)
        rule_preds.append(rule_label)

        label = f"{model_label} / {rule_label}"
        annotated = draw_skeleton(frame.copy(), keypoints['output_0'].numpy(), label)
        output_frames.append(annotated)

    cap.release()

    # 📊 Stats
    deduped = deduplicate_punches(rule_preds)
    total_punches = len(deduped)
    duration = len(rule_preds) / fps
    rate = total_punches / duration

    stats = {
        "total_punches": total_punches,
        "duration_seconds": duration,
        "punch_rate": rate,
        "fps": fps,
        "frame_count": len(rule_preds)
    }

    st.subheader("🥊 Refined Punch Stats")
    st.write(f"✅ Unique Punches: {total_punches}")
    st.write(f"⏱️ Duration: {duration:.2f} sec")
    st.write(f"⚡ Rate: {rate:.2f} punches/sec (~{rate * 60:.1f} per min)")

    return output_frames, model_preds, rule_preds, fps, stats, width, height

# 🖥️ GUI
st.title("🥊 Punch Detection: Classifier vs MoveNet Rule-Based")

uploaded_model = st.file_uploader("Upload Trained Classifier (.joblib)", type=["joblib"])
clf = None
if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_model:
        tmp_model.write(uploaded_model.read())
        tmp_model.flush()
        clf = joblib.load(tmp_model.name)

model = load_movenet_model()

uploaded_file = st.file_uploader("Upload Boxing Video", type=["mp4", "avi", "mov"])
if uploaded_file and clf:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.info("⏳ Processing video and predicting punches...")
    frames, preds_model, preds_rule, fps, stats, width, height = extract_and_predict(video_path, model, clf)

    raw_path = os.path.join(temp_dir, "raw_output.mp4")
    final_path = os.path.join(temp_dir, "predicted_output.mp4")

    save_video(frames, fps, width, height, raw_path)
    reencode_with_ffmpeg(raw_path, final_path)

    st.success("✅ Prediction complete! Showing result:")
    with open(final_path, 'rb') as f:
        st.video(f.read())

    # Comparison CSV
    df_compare = pd.DataFrame({
        'frame': list(range(len(preds_model))),
        'model_prediction': preds_model,
        'movenet_prediction': preds_rule
    })
    comp_path = os.path.join(temp_dir, "punch_comparison.csv")
    df_compare.to_csv(comp_path, index=False)
    st.download_button("📥 Download Comparison CSV", data=open(comp_path, "rb").read(), file_name="punch_comparison.csv", mime="text/csv")

    # Classifier prediction CSV
    df_model = pd.DataFrame({
        'frame': list(range(len(preds_model))),
        'model_prediction': preds_model
    })
    model_path = os.path.join(temp_dir, "model_predictions.csv")
    df_model.to_csv(model_path, index=False)
    st.download_button("📥 Download Classifier Predictions CSV", data=open(model_path, "rb").read(), file_name="model_predictions.csv", mime="text/csv")

    # Filtered CSV
    none_count = preds_rule.count("none")
    filtered = [(i, p) for i, p in enumerate(preds_rule) if p != "none"]
    frames_filt = [i for i, _ in filtered]
    labels_filt = [p for _, p in filtered]

    st.write(f"🚫 'none' labels: {none_count}")
    st.write(f"✅ Filtered rows: {len(labels_filt)}")

    df_filtered = pd.DataFrame({
        "frame": frames_filt,
        "movenet_prediction": labels_filt
    })

    filtered_path = os.path.join(temp_dir, "movenet_punches.csv")
    df_filtered.to_csv(filtered_path, index=False)

    with open(filtered_path, "r") as f_check:
        line_count = sum(1 for _ in f_check) - 1
    st.write(f"📄 Final filtered CSV rows: {line_count}")

    st.download_button("📥 Download MoveNet Filtered CSV", data=open(filtered_path, "rb").read(), file_name="movenet_punches.csv", mime="text/csv")

    # Agreement summary
    st.subheader("📊 Prediction Comparison Summary")
    agree = sum([m == r for m, r in zip(preds_model, preds_rule)])
    st.write(f"✅ Agreement: {agree}/{len(preds_model)} frames ({agree / len(preds_model) * 100:.2f}%)")

    st.dataframe(df_compare.head(10))

# ✅ Requirements
with open("requirements.txt", "w") as f:
    f.write('''streamlit
tensorflow
tensorflow_hub
opencv-python-headless
pandas
numpy
scikit-learn
joblib
ffmpeg-python
''')
print("✅ requirements.txt saved")
