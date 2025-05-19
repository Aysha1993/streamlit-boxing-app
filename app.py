import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import ffmpeg
# import io # Not used directly, can be removed if not needed elsewhere
# import json # Not used directly in the final version of expand_keypoints, but kept if other parts might use it.
import tensorflow as tf
import tensorflow_hub as hub
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Streamlit setup
st.set_option('client.showErrorDetails', True)
st.title("🥊 Boxing Analyzer App")

# Define expected number of features globally or retrieve from model meta-data
EXPECTED_FEATURES = 51 # 17 keypoints * 3 (y, x, confidence)

# Load MoveNet MultiPose model from TFHub
@st.cache_resource
def load_movenet_model(): # Renamed for clarity
    os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'
    return hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")

movenet = load_movenet_model() # Name it movenet for clarity

# Utility functions
def extract_keypoints(results):
    people = []
    raw_persons_data = results['output_0'].numpy()[0]
    for person_data in raw_persons_data:
        keypoints_coords_scores = np.array(person_data[:51]).reshape(17, 3)
        person_score = person_data[55]
        if person_score > 0.2 and np.mean(keypoints_coords_scores[:, 2]) > 0.1:
            people.append(keypoints_coords_scores.tolist())
    return people


def classify_punch(keypoints_list_of_lists):
    result = []
    for kp_person in keypoints_list_of_lists:
        lw, rw = kp_person[9], kp_person[10]
        ls, rs = kp_person[5], kp_person[6]
        le, re = kp_person[7], kp_person[8]
        punch_type = "Guard"
        if lw[2] > 0.2 and ls[2] > 0.2 and le[2] > 0.2:
            if lw[1] < ls[1] and abs(lw[0] - ls[0]) < abs(lw[1]-le[1])*0.5 :
                punch_type = "Left Jab"
            elif abs(lw[1] - ls[1]) > 0.1 and abs(lw[0] - le[0]) > abs(lw[1]-le[1])*0.5 :
                punch_type = "Left Cross"
            elif abs(le[0] - lw[0]) < abs(le[1]-lw[1])*0.5 and le[0] < ls[0]:
                 punch_type = "Left Hook"
        if rw[2] > 0.2 and rs[2] > 0.2 and re[2] > 0.2:
            if rw[1] > rs[1] and abs(rw[0] - rs[0]) < abs(rw[1]-re[1])*0.5:
                punch_type = "Right Jab"
            elif abs(rw[1] - rs[1]) > 0.1 and abs(rw[0] - re[0]) > abs(rw[1]-re[1])*0.5 :
                punch_type = "Right Cross"
            elif abs(re[0] - rw[0]) < abs(re[1]-rw[1])*0.5 and re[0] < rs[0]:
                punch_type = "Right Hook"
        result.append(punch_type)
    return result

def check_posture(keypoints_list_of_lists):
    feedback = []
    for kp_person in keypoints_list_of_lists:
        msgs = []
        if kp_person[7][2] > 0.2 and kp_person[11][2] > 0.2 and kp_person[7][0] > kp_person[11][0]: msgs.append("Left Elbow drop relative to hip")
        if kp_person[8][2] > 0.2 and kp_person[12][2] > 0.2 and kp_person[8][0] > kp_person[12][0]: msgs.append("Right Elbow drop relative to hip")
        if kp_person[5][2] > 0.2 and kp_person[11][2] > 0.2 and kp_person[5][0] > kp_person[11][0]: msgs.append("Left Shoulder drop relative to hip")
        if kp_person[6][2] > 0.2 and kp_person[12][2] > 0.2 and kp_person[6][0] > kp_person[12][0]: msgs.append("Right Shoulder drop relative to hip")
        if kp_person[15][2] > 0.2 and kp_person[13][2] > 0.2 and kp_person[15][0] < kp_person[13][0] - 0.05 : msgs.append("Left Knee Bent")
        if kp_person[16][2] > 0.2 and kp_person[14][2] > 0.2 and kp_person[16][0] < kp_person[14][0] - 0.05 : msgs.append("Right Knee Bent")
        if kp_person[9][2] > 0.2 and kp_person[7][2] > 0.2 and kp_person[9][0] > kp_person[7][0]: msgs.append("Left Wrist drop")
        if kp_person[10][2] > 0.2 and kp_person[8][2] > 0.2 and kp_person[10][0] > kp_person[8][0]: msgs.append("Right Wrist drop")
        if not msgs:
            msgs.append("Good Posture")
        feedback.append(", ".join(msgs))
    return feedback

def detect_gloves(keypoints_list_of_lists, distance_thresh=0.05):
    gloves_status = []
    for kp_person in keypoints_list_of_lists:
        lw, le = kp_person[9], kp_person[7]
        rw, re = kp_person[10], kp_person[8]
        def is_glove_present(wrist, elbow):
            if wrist[2] > 0.2 and elbow[2] > 0.2:
                dist = np.linalg.norm(np.array(wrist[:2]) - np.array(elbow[:2]))
                return dist > distance_thresh
            return False
        left_glove = "yes" if is_glove_present(lw, le) else "no"
        right_glove = "yes" if is_glove_present(rw, re) else "no"
        gloves_status.append(f"Gloves: L-{left_glove} R-{right_glove}")
    return gloves_status

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_annotations(frame, all_persons_keypoints, punches, postures, gloves_statuses):
    h, w = frame.shape[:2]
    for i, person_kps in enumerate(all_persons_keypoints):
        for (y, x, score) in person_kps:
            if score > 0.2:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)
        for (p1_idx, p2_idx) in SKELETON_EDGES:
            y1, x1, s1 = person_kps[p1_idx]
            y2, x2, s2 = person_kps[p2_idx]
            if s1 > 0.2 and s2 > 0.2:
                cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (255, 0, 0), 2)
        nose_y, nose_x, nose_s = person_kps[0]
        text_x = int(nose_x * w) + 10 if nose_s > 0.2 else 20
        text_y_start = int(nose_y * h) if nose_s > 0.2 else h - 50
        if i < len(punches) and punches[i]:
            cv2.putText(frame, f"P{i}: {punches[i]}", (text_x, text_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if i < len(postures) and postures[i]:
             cv2.putText(frame, f"P{i} Posture: {postures[i]}", (text_x, text_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if i < len(gloves_statuses) and gloves_statuses[i]:
            cv2.putText(frame, f"P{i} {gloves_statuses[i]}", (text_x, text_y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame

def expand_keypoints(keypoints_one_person):
    if isinstance(keypoints_one_person, str):
        try: keypoints_one_person = json.loads(keypoints_one_person) # json might still be needed if stringified kps are passed
        except json.JSONDecodeError: return pd.Series(dtype='float64')
    if not isinstance(keypoints_one_person, list) or \
       not all(isinstance(kp, (list, tuple)) and len(kp) == 3 for kp in keypoints_one_person) or \
       len(keypoints_one_person) != 17:
        return pd.Series(dtype='float64')
    try:
        data = {}
        for i, kp in enumerate(keypoints_one_person):
            data[f'y_{i}'] = kp[0]; data[f'x_{i}'] = kp[1]; data[f's_{i}'] = kp[2]
        return pd.Series(data)
    except Exception: return pd.Series(dtype='float64')

def flatten_keypoints(keypoints_one_person):
    if isinstance(keypoints_one_person, list) and \
       len(keypoints_one_person) == 17 and \
       all(isinstance(kp, (list, tuple)) and len(kp) == 3 for kp in keypoints_one_person):
        return [v for kp in keypoints_one_person for v in kp]
    return []

if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False; st.session_state.svm_model = None
    st.session_state.label_encoder = None; st.session_state.expected_features = EXPECTED_FEATURES

uploaded_files = st.file_uploader("Upload multiple boxing videos for Training/Processing", type=["mp4", "avi", "mov"], accept_multiple_files=True)

if uploaded_files:
    all_logs_for_training = []
    progress_bar = st.progress(0)
    for video_idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"📦 Processing for training data: {uploaded_file.name}")
        temp_dir_train = tempfile.mkdtemp()
        input_path_train = os.path.join(temp_dir_train, uploaded_file.name)
        with open(input_path_train, 'wb') as f: f.write(uploaded_file.read())
        cap = cv2.VideoCapture(input_path_train)
        width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
        raw_output_train_path = os.path.join(temp_dir_train, "raw_output_train.mp4")
        out_writer = cv2.VideoWriter(raw_output_train_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        current_video_punch_log = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: st.warning(f"Could not get total frames for {uploaded_file.name}. Skipping."); continue
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            resized_frame = cv2.resize(frame, (256, 256))
            input_tensor = tf.cast(tf.convert_to_tensor(resized_frame[None, ...]), dtype=tf.int32)
            results = movenet.signatures['serving_default'](input_tensor)
            keypoints_all_persons = extract_keypoints(results)
            if not keypoints_all_persons:
                out_writer.write(frame); frame_idx += 1
                if frame_idx % 10 == 0: progress_bar.progress(min(((video_idx + (frame_idx / total_frames)) / len(uploaded_files)), 1.0))
                continue
            punches = classify_punch(keypoints_all_persons)
            postures = check_posture(keypoints_all_persons)
            gloves_statuses = detect_gloves(keypoints_all_persons)
            annotated_frame = draw_annotations(frame.copy(), keypoints_all_persons, punches, postures, gloves_statuses)
            out_writer.write(annotated_frame)
            for i, person_kps in enumerate(keypoints_all_persons):
                current_video_punch_log.append({
                    "video": uploaded_file.name, "frame": frame_idx, "person_id_in_frame": i,
                    "punch": punches[i] if i < len(punches) else "N/A",
                    "posture": postures[i] if i < len(postures) else "N/A",
                    "gloves": gloves_statuses[i] if i < len(gloves_statuses) else "N/A",
                    "keypoints_for_person": person_kps
                })
            frame_idx += 1
            if frame_idx % 10 == 0: progress_bar.progress(min(((video_idx + (frame_idx / total_frames)) / len(uploaded_files)), 1.0))
        cap.release(); out_writer.release()
        final_output_train_path = os.path.join(temp_dir_train, f"final_train_{uploaded_file.name}")
        try:
            ffmpeg.input(raw_output_train_path).output(final_output_train_path, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            st.video(final_output_train_path)
            with open(final_output_train_path, "rb") as f_video: st.download_button(f"📥 Download Annotated Training Video: {uploaded_file.name}", f_video, file_name=f"annotated_train_{uploaded_file.name}", mime="video/mp4")
        except ffmpeg.Error as e: st.error(f"FFmpeg error (training video): {e.stderr.decode('utf8') if e.stderr else 'Unknown'}")
        df_video_log = pd.DataFrame(current_video_punch_log)
        if not df_video_log.empty:
            if 'keypoints_for_person' in df_video_log.columns and not df_video_log['keypoints_for_person'].dropna().empty:
                 st.json(df_video_log['keypoints_for_person'].dropna().iloc[0])
            expanded_df_video = df_video_log.copy()
            if 'keypoints_for_person' in expanded_df_video.columns:
                keypoint_features_df = expanded_df_video['keypoints_for_person'].apply(expand_keypoints)
                expanded_df_video = pd.concat([expanded_df_video.drop(columns=['keypoints_for_person']), keypoint_features_df], axis=1) if not keypoint_features_df.empty else expanded_df_video.drop(columns=['keypoints_for_person'])
                st.dataframe(expanded_df_video.head())
                st.download_button(f"📄 Download Log CSV for {uploaded_file.name}", expanded_df_video.to_csv(index=False).encode('utf-8'), file_name=f"log_{uploaded_file.name}.csv", mime="text/csv")
            all_logs_for_training.extend(current_video_punch_log)
    progress_bar.empty()

    if all_logs_for_training:
        df_all_logs = pd.DataFrame(all_logs_for_training)
        if 'keypoints_for_person' not in df_all_logs.columns or df_all_logs['keypoints_for_person'].isnull().all():
            st.error("Error: 'keypoints_for_person' data is missing. Cannot train models.")
        else:
            df_all_logs["flat_kp"] = df_all_logs["keypoints_for_person"].apply(flatten_keypoints)
            df_all_logs = df_all_logs[df_all_logs["flat_kp"].apply(lambda x: len(x) == EXPECTED_FEATURES)]
            if df_all_logs.empty:
                st.error("No valid keypoint data for training after flattening.")
            else:
                X = np.vstack(df_all_logs["flat_kp"].values)
                y = df_all_logs["punch"].values
                if len(np.unique(y)) < 2 and len(y) >0 : # Check if less than 2 unique classes AND y is not empty
                    st.error(f"Not enough class diversity for training. Found only {len(np.unique(y))} unique punch type(s) ('{np.unique(y)[0]}' if only one). Need at least 2 distinct classes for meaningful training and evaluation.")
                elif len(y) == 0:
                     st.error("No punch labels (y) available for training. Cannot proceed.")
                else: # Sufficient overall class diversity initially assumed here, specific class counts checked next
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    
                    unique_classes_encoded, class_counts_encoded = np.unique(y_encoded, return_counts=True)
                    min_samples_in_any_class = 0
                    problematic_class_name_str = "N/A"

                    if len(class_counts_encoded) > 0:
                        min_samples_in_any_class = class_counts_encoded.min()
                        if 0 < min_samples_in_any_class < 2 : # If smallest class has 1 sample
                             problematic_class_idx = class_counts_encoded.argmin()
                             problematic_class_encoded_val = unique_classes_encoded[problematic_class_idx]
                             problematic_class_name_str = le.classes_[problematic_class_encoded_val]

                    can_stratify = True
                    if len(unique_classes_encoded) > 1: # More than one class overall
                        if min_samples_in_any_class < 2: # But at least one class has too few samples
                            can_stratify = False
                            st.warning(
                                f"Stratification has been disabled for train/test split. "
                                f"The least populated class ('{problematic_class_name_str}') "
                                f"has only {min_samples_in_any_class} sample(s). "
                                f"A minimum of 2 samples per class is needed for stratification."
                            )
                    else: # Only one class (or zero classes if y_encoded was empty)
                        can_stratify = False
                        if len(unique_classes_encoded) == 1:
                             st.info("Only one class present in the data. Stratification is not applicable for train/test split.")
                    
                    # Ensure X and y_encoded are not empty before splitting
                    if X.shape[0] > 0 and y_encoded.shape[0] > 0 and X.shape[0] == y_encoded.shape[0]:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_encoded, 
                            test_size=0.2, 
                            random_state=42, 
                            stratify=y_encoded if can_stratify else None
                        )

                        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                            st.error("Training or testing set became empty after split. This can happen with very small datasets. Cannot train models.")
                        else:
                            svm_model_trained = svm.SVC(kernel='linear', probability=True)
                            svm_model_trained.fit(X_train, y_train)
                            tree_model_trained = DecisionTreeClassifier(max_depth=5, random_state=42)
                            tree_model_trained.fit(X_train, y_train)
                            svm_preds = svm_model_trained.predict(X_test)
                            tree_preds = tree_model_trained.predict(X_test)
                            st.write("### 📊 Model Evaluation on Training Data")
                            st.write(f"🔹 SVM Accuracy: {accuracy_score(y_test, svm_preds):.2f}")
                            st.text(classification_report(y_test, svm_preds, target_names=le.classes_, zero_division=0))
                            st.write(f"🔹 Decision Tree Accuracy: {accuracy_score(y_test, tree_preds):.2f}")
                            st.text(classification_report(y_test, tree_preds, target_names=le.classes_, zero_division=0))
                            fig1, ax1 = plt.subplots(); ConfusionMatrixDisplay(confusion_matrix(y_test, svm_preds), display_labels=le.classes_).plot(ax=ax1, xticks_rotation='vertical'); st.pyplot(fig1)
                            fig2, ax2 = plt.subplots(); ConfusionMatrixDisplay(confusion_matrix(y_test, tree_preds), display_labels=le.classes_).plot(ax=ax2, xticks_rotation='vertical'); st.pyplot(fig2)
                            st.session_state.svm_model = svm_model_trained
                            st.session_state.label_encoder = le
                            st.session_state.model_ready = True
                            st.session_state.expected_features = X_train.shape[1] 
                            st.success("✅ SVM and Decision Tree models trained and ready for prediction.")
                    else:
                        st.error("Not enough data to perform train/test split (X or y is empty, or lengths mismatch).")
    else:
        st.info("No training data generated. Cannot train models.")

st.write("---"); st.header("🎬 Predict on a New Clip")
video_file_predict = st.file_uploader("Upload a test video for prediction", type=["mp4", "mov", "avi"], key="predict_uploader")
if video_file_predict is not None:
    if not st.session_state.model_ready or st.session_state.svm_model is None or st.session_state.label_encoder is None:
        st.error("🚫 Models are not trained/loaded. Process training videos first.")
    else:
        svm_model_predict = st.session_state.svm_model; le_predict = st.session_state.label_encoder
        st.info(f"👁️ Processing for prediction: {video_file_predict.name}")
        temp_dir_predict = tempfile.mkdtemp()
        input_path_predict = os.path.join(temp_dir_predict, "test_video.mp4")
        with open(input_path_predict, "wb") as f: f.write(video_file_predict.read())
        cap_predict = cv2.VideoCapture(input_path_predict)
        width_p, height_p, fps_p = int(cap_predict.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_predict.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap_predict.get(cv2.CAP_PROP_FPS)
        raw_output_predict_path = os.path.join(temp_dir_predict, "raw_output_predict.mp4")
        out_predict_writer = cv2.VideoWriter(raw_output_predict_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_p, (width_p, height_p))
        stframe_predict = st.empty(); frame_count_predict = 0
        total_frames_predict = int(cap_predict.get(cv2.CAP_PROP_FRAME_COUNT))
        predict_progress = st.progress(0)
        while cap_predict.isOpened():
            ret, frame = cap_predict.read()
            if not ret: break
            current_frame_for_prediction = frame.copy()
            try:
                resized_predict = cv2.resize(current_frame_for_prediction, (256, 256))
                input_tensor_predict = tf.cast(tf.convert_to_tensor(resized_predict[None, ...]), dtype=tf.int32)
                results_predict = movenet.signatures['serving_default'](input_tensor_predict)
                all_persons_kps_predict = extract_keypoints(results_predict)
                if not all_persons_kps_predict:
                    cv2.putText(current_frame_for_prediction, "No person detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    for person_kps_item in all_persons_kps_predict:
                        for (p1_idx, p2_idx) in SKELETON_EDGES:
                            y1, x1, s1 = person_kps_item[p1_idx]; y2, x2, s2 = person_kps_item[p2_idx]
                            if s1 > 0.2 and s2 > 0.2: cv2.line(current_frame_for_prediction, (int(x1*width_p),int(y1*height_p)), (int(x2*width_p),int(y2*height_p)), (255,100,0),2)
                        for (y,x,s) in person_kps_item:
                             if s>0.2: cv2.circle(current_frame_for_prediction, (int(x*width_p),int(y*height_p)),3,(0,255,0),-1)
                    for person_idx, person_kps in enumerate(all_persons_kps_predict):
                        if not person_kps or len(person_kps) != 17: continue
                        flat_kp_predict = flatten_keypoints(person_kps)
                        if len(flat_kp_predict) != st.session_state.expected_features:
                            cv2.putText(current_frame_for_prediction, f"P{person_idx}: Feat Error (len {len(flat_kp_predict)})", (30,40+person_idx*60), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                            continue
                        X_input_predict = np.array(flat_kp_predict).reshape(1, -1)
                        pred_class_svm = svm_model_predict.predict(X_input_predict)
                        predicted_label_svm = le_predict.inverse_transform(pred_class_svm)[0]
                        confidences_person = [kp[2] for kp in person_kps if isinstance(kp,(list,tuple)) and len(kp)==3]; avg_conf_person = np.mean(confidences_person) if confidences_person else 0.0
                        nose_y,nose_x,nose_s = person_kps[0]
                        text_x_coord = int(nose_x*width_p)+10 if nose_s>0.2 else (30+person_idx*150); text_y_start_coord = int(nose_y*height_p)-20 if nose_s>0.2 else 40
                        cv2.putText(current_frame_for_prediction, f"P{person_idx} Punch: {predicted_label_svm}", (text_x_coord,text_y_start_coord), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                        cv2.putText(current_frame_for_prediction, f"Conf: {avg_conf_person:.2f}", (text_x_coord,text_y_start_coord+25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            except Exception as e: st.error(f"⚠️ Frame {frame_count_predict} prediction error: {str(e)}")
            out_predict_writer.write(current_frame_for_prediction)
            stframe_predict.image(cv2.cvtColor(current_frame_for_prediction,cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count_predict+1}", use_container_width=True)
            frame_count_predict += 1
            if total_frames_predict > 0 : predict_progress.progress(frame_count_predict / total_frames_predict)
        cap_predict.release(); out_predict_writer.release(); predict_progress.empty()
        final_output_predict_path = os.path.join(temp_dir_predict, f"annotated_pred_{video_file_predict.name}")
        try:
            ffmpeg.input(raw_output_predict_path).output(final_output_predict_path,vcodec='libx264',acodec='aac',strict='experimental').run(overwrite_output=True,capture_stdout=True,capture_stderr=True)
            st.success(f"✅ Annotated prediction video ready: {video_file_predict.name}")
            st.video(final_output_predict_path)
            with open(final_output_predict_path, "rb") as f_pred_video: st.download_button(f"📥 Download Clip: {video_file_predict.name}", f_pred_video, file_name=f"annotated_pred_{video_file_predict.name}", mime="video/mp4")
        except ffmpeg.Error as e: st.error(f"FFmpeg error (prediction video): {e.stderr.decode('utf8') if e.stderr else 'Unknown'}")

requirements = '''streamlit
tensorflow
tensorflow_hub
opencv-python-headless
pandas
numpy
scikit-learn
joblib
ffmpeg-python
matplotlib
'''
with open("requirements.txt", "w") as f: f.write(requirements)
print("✅ requirements.txt saved")
