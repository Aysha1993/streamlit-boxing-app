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

# Clean up old temp files at the start
temp_dir = tempfile.gettempdir()
for f in ["movenet_punches.csv", "punch_comparison.csv", "raw_output.mp4", "predicted_output.mp4"]:
    try:
        os.remove(os.path.join(temp_dir, f))
    except FileNotFoundError:
        pass

# Load MoveNet
@st.cache_resource
def load_movenet_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    return model.signatures['serving_default']

def preprocess_keypoints(keypoints):
    return keypoints[0, 0, :, :3].flatten()

def rule_based_prediction(flat_kp):
    kp = np.array(flat_kp).reshape(17, 3)
    if kp[9][1] < kp[7][1]:
        return "Jab"
    elif kp[10][1] < kp[8][1]:
        return "Cross"
    return "none"

def draw_skeleton(frame, keypoints, label=None):
    keypoints = keypoints[0, 0, :, :2]
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    if label:
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    return frame

def save_video(frames, fps, width, height, output_path):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def reencode_with_ffmpeg(input_path, output_path):
    ffmpeg.input(input_path).output(output_path, vcodec='libx264', acodec='aac', pix_fmt='yuv420p').run(overwrite_output=True)

def deduplicate_punches(preds):
    result, last = [], "none"
    for p in preds:
        if p != "none" and p != last:
            result.append(p)
        last = p
    return result

def extract_and_predict(video_path, model, clf):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))

    frames, preds_model, preds_rule = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
        input_tensor = tf.cast(img, dtype=tf.int32)
        kp = model(input_tensor)['output_0'].numpy()
        flat_kp = preprocess_keypoints(kp)

        model_label = clf.predict([flat_kp])[0]
        rule_label = rule_based_prediction(flat_kp)

        preds_model.append(model_label)
        preds_rule.append(rule_label)

        frames.append(draw_skeleton(frame.copy(), kp, f"{model_label}/{rule_label}"))

    cap.release()

    return frames, preds_model, preds_rule, fps, width, height

# -------------------- Streamlit GUI --------------------
st.title("🥊 Punch Detection: MoveNet + Classifier")

uploaded_model = st.file_uploader("Upload Trained Classifier (.joblib)", type=["joblib"])
clf = None
if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        tmp.write(uploaded_model.read())
        tmp.flush()
        clf = joblib.load(tmp.name)

model = load_movenet_model()

uploaded_file = st.file_uploader("Upload Boxing Video", type=["mp4", "avi", "mov"])
if uploaded_file and clf:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.info("⏳ Processing...")
    frames, preds_model, preds_rule, fps, width, height = extract_and_predict(video_path, model, clf)

    raw_path = os.path.join(temp_dir, "raw_output.mp4")
    final_path = os.path.join(temp_dir, "predicted_output.mp4")

    save_video(frames, fps, width, height, raw_path)
    reencode_with_ffmpeg(raw_path, final_path)

    st.success("✅ Done! Showing video:")
    st.video(open(final_path, "rb").read())

    # --- Save Comparison CSV ---
    df_comparison = pd.DataFrame({
        "frame": list(range(len(preds_model))),
        "model_prediction": preds_model,
        "movenet_prediction": preds_rule
    })
    comp_path = os.path.join(temp_dir, "punch_comparison.csv")
    df_comparison.to_csv(comp_path, index=False)
    st.download_button("📥 Download Full Comparison CSV", data=open(comp_path, "rb"), file_name="punch_comparison.csv")

    # --- Filtered MoveNet Only CSV ---
    filtered = [(i, p) for i, p in enumerate(preds_rule) if p != "none"]
    filtered_frames = [i for i, _ in filtered]
    filtered_labels = [p for _, p in filtered]

    st.write(f"🚫 'none' labels: {preds_rule.count('none')}")
    st.write(f"✅ Filtered rows: {len(filtered_labels)}")

    df_filtered = pd.DataFrame({
        "frame": filtered_frames,
        "movenet_prediction": filtered_labels
    })

    movenet_path = os.path.join(temp_dir, "movenet_punches.csv")
    df_filtered.to_csv(movenet_path, index=False)

    with open(movenet_path, "r") as f_check:
        lines = sum(1 for _ in f_check) - 1
    st.write(f"📄 Final filtered CSV rows: {lines}")

    with open(movenet_path, "rb") as f:
        st.download_button("📥 Download MoveNet Filtered CSV", data=f, file_name="movenet_punches.csv")

# Save requirements.txt
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
