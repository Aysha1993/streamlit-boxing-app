import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import ffmpeg
import io
import json
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
    # Output tensor for MultiPose Lightning model has shape [1, 6, 56]
    # It contains information for up to 6 people.
    # Each person: [y, x, score] for 17 keypoints (17*3=51 values), 
    # then [ymin, xmin, ymax, xmax, score] for the bounding box (5 values). Total 51+5 = 56.
    raw_persons_data = results['output_0'].numpy()[0]  # Shape (6, 56)
    for person_data in raw_persons_data:
        keypoints_coords_scores = np.array(person_data[:51]).reshape(17, 3) # y, x, score for each keypoint
        person_score = person_data[55] # Overall detection score for this person instance

        # Filter based on overall person score and average keypoint confidence
        # Adjusted threshold for keypoint confidence to be more robust
        if person_score > 0.2 and np.mean(keypoints_coords_scores[:, 2]) > 0.1: # Check mean of keypoint scores
            people.append(keypoints_coords_scores.tolist())
    return people


def classify_punch(keypoints_list_of_lists): # Parameter renamed for clarity
    result = []
    for kp_person in keypoints_list_of_lists: # kp_person is a list of 17 keypoints for one person
        lw, rw = kp_person[9], kp_person[10] # Left Wrist, Right Wrist
        ls, rs = kp_person[5], kp_person[6] # Left Shoulder, Right Shoulder
        le, re = kp_person[7], kp_person[8] # Left Elbow, Right Elbow

        # Ensure keypoints have high enough confidence (score is kp[2])
        # Example: Check confidence of wrist and shoulder for jab
        punch_type = "Guard" # Default
        if lw[2] > 0.2 and ls[2] > 0.2 and le[2] > 0.2: # Left arm keypoints confidence
            if lw[1] < ls[1] and abs(lw[0] - ls[0]) < abs(lw[1]-le[1])*0.5 : # lw[1] is x-coord, lw[0] is y-coord. Jab: wrist x < shoulder x
                punch_type = "Left Jab"
            elif abs(lw[1] - ls[1]) > 0.1 and abs(lw[0] - le[0]) > abs(lw[1]-le[1])*0.5 : # Cross: significant horizontal distance
                punch_type = "Left Cross"
            elif abs(le[0] - lw[0]) < abs(le[1]-lw[1])*0.5 and le[0] < ls[0]: # Hook: elbow and wrist y are somewhat aligned, elbow higher or at shoulder level
                 punch_type = "Left Hook"


        if rw[2] > 0.2 and rs[2] > 0.2 and re[2] > 0.2: # Right arm keypoints confidence
            if rw[1] > rs[1] and abs(rw[0] - rs[0]) < abs(rw[1]-re[1])*0.5: # Right Jab: wrist x > shoulder x (assuming standard boxing stance view)
                                                            # This logic for Jab (rw[0] < rs[0]) was likely for y-coordinates if fighter is sideways
                                                            # Assuming fighter is somewhat facing, x-coordinate difference is more telling for reach
                punch_type = "Right Jab" # If left was also a punch, this might override. Consider logic for simultaneous if needed.
            elif abs(rw[1] - rs[1]) > 0.1 and abs(rw[0] - re[0]) > abs(rw[1]-re[1])*0.5 :
                punch_type = "Right Cross"
            elif abs(re[0] - rw[0]) < abs(re[1]-rw[1])*0.5 and re[0] < rs[0]:
                punch_type = "Right Hook"
        
        result.append(punch_type)
    return result

def check_posture(keypoints_list_of_lists): # Parameter renamed
    feedback = []
    for kp_person in keypoints_list_of_lists:
        msgs = []
        # y-coordinates are kp[0], x-coordinates are kp[1]
        # Assuming normalized coordinates where (0,0) is top-left.
        # Higher y means lower on frame.
        # For "drop", it usually means the joint is lower than it should be.
        # Example: Elbow drop might mean elbow y > shoulder y when it shouldn't be.
        # These rules need careful validation based on expected posture and coordinate system.
        if kp_person[7][2] > 0.2 and kp_person[11][2] > 0.2 and kp_person[7][0] > kp_person[11][0]: msgs.append("Left Elbow drop relative to hip") # Left Elbow y > Left Hip y
        if kp_person[8][2] > 0.2 and kp_person[12][2] > 0.2 and kp_person[8][0] > kp_person[12][0]: msgs.append("Right Elbow drop relative to hip")
        if kp_person[5][2] > 0.2 and kp_person[11][2] > 0.2 and kp_person[5][0] > kp_person[11][0]: msgs.append("Left Shoulder drop relative to hip")
        if kp_person[6][2] > 0.2 and kp_person[12][2] > 0.2 and kp_person[6][0] > kp_person[12][0]: msgs.append("Right Shoulder drop relative to hip")
        # Knee bent: knee y is significantly higher (more bent) than ankle y, or knee x is forward.
        # This logic kp[15][0] < kp[13][0] - 0.05 suggests left ankle y < left knee y - offset
        if kp_person[15][2] > 0.2 and kp_person[13][2] > 0.2 and kp_person[15][0] < kp_person[13][0] - 0.05 : msgs.append("Left Knee Bent")
        if kp_person[16][2] > 0.2 and kp_person[14][2] > 0.2 and kp_person[16][0] < kp_person[14][0] - 0.05 : msgs.append("Right Knee Bent")
        # Wrist drop: wrist y > elbow y when arm is extended or in guard.
        if kp_person[9][2] > 0.2 and kp_person[7][2] > 0.2 and kp_person[9][0] > kp_person[7][0]: msgs.append("Left Wrist drop")
        if kp_person[10][2] > 0.2 and kp_person[8][2] > 0.2 and kp_person[10][0] > kp_person[8][0]: msgs.append("Right Wrist drop")
        
        if not msgs:
            msgs.append("Good Posture")
        feedback.append(", ".join(msgs))
    return feedback

def detect_gloves(keypoints_list_of_lists, distance_thresh=0.05): # Adjusted threshold for normalized coords
    gloves_status = []
    for kp_person in keypoints_list_of_lists:
        lw, le = kp_person[9], kp_person[7] # Left Wrist, Left Elbow
        rw, re = kp_person[10], kp_person[8] # Right Wrist, Right Elbow

        def is_glove_present(wrist, elbow):
            if wrist[2] > 0.2 and elbow[2] > 0.2: # Check confidence
                # Distance between wrist and elbow (using y,x which are kp[0], kp[1])
                dist = np.linalg.norm(np.array(wrist[:2]) - np.array(elbow[:2]))
                # This logic is a bit indirect for glove detection.
                # A more direct method might involve looking for a blob near the hand or color.
                # This current logic might indicate if the forearm is visible / extended.
                return dist > distance_thresh # If wrist and elbow are far apart enough
            return False

        left_glove = "yes" if is_glove_present(lw, le) else "no" # Placeholder logic
        right_glove = "yes" if is_glove_present(rw, re) else "no" # Placeholder logic
        gloves_status.append(f"Gloves: L-{left_glove} R-{right_glove}")
    return gloves_status


SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Torso and Arms
    (5, 11), (6, 12), (11, 12), # Pelvis
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]

def draw_annotations(frame, all_persons_keypoints, punches, postures, gloves_statuses): # Renamed params
    h, w = frame.shape[:2]

    for i, person_kps in enumerate(all_persons_keypoints): # Iterate through each person
        # Draw keypoints
        for (y, x, score) in person_kps: # y,x are normalized
            if score > 0.2:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Draw skeleton
        for (p1_idx, p2_idx) in SKELETON_EDGES:
            y1, x1, s1 = person_kps[p1_idx]
            y2, x2, s2 = person_kps[p2_idx]
            if s1 > 0.2 and s2 > 0.2:
                pt1 = int(x1 * w), int(y1 * h)
                pt2 = int(x2 * w), int(y2 * h)
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        # Glove annotation (simplified, using wrist position for text)
        for side, wrist_idx in zip(["L", "R"], [9, 10]):
            y, x, s = person_kps[wrist_idx]
            if s > 0.2:
                cx, cy = int(x * w), int(y * h)
                pad = 15
                # Simple text for glove status if available
                # cv2.rectangle(frame, (cx - pad, cy - pad), (cx + pad, cy + pad), (0, 0, 255), 2)
                # cv2.putText(frame, f"{side} Glove?", (cx - pad, cy - pad - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


        # Text annotations for punch, posture
        # Find a point to anchor text for this person (e.g., near the nose or neck)
        nose_y, nose_x, nose_s = person_kps[0] # Nose keypoint
        text_x = int(nose_x * w) + 10 if nose_s > 0.2 else 20 # Fallback x
        text_y_start = int(nose_y * h) if nose_s > 0.2 else h - 50 # Fallback y
        
        if i < len(punches) and punches[i]:
            cv2.putText(frame, f"P{i}: {punches[i]}", (text_x, text_y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if i < len(postures) and postures[i]:
             cv2.putText(frame, f"P{i} Posture: {postures[i]}", (text_x, text_y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if i < len(gloves_statuses) and gloves_statuses[i]:
            cv2.putText(frame, f"P{i} {gloves_statuses[i]}", (text_x, text_y_start + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def expand_keypoints(keypoints_one_person): # Renamed
    if isinstance(keypoints_one_person, str):
        try:
            keypoints_one_person = json.loads(keypoints_one_person)
        except json.JSONDecodeError:
            return pd.Series(dtype='float64') # Return empty series with dtype
    
    if not isinstance(keypoints_one_person, list) or \
       not all(isinstance(kp, (list, tuple)) and len(kp) == 3 for kp in keypoints_one_person) or \
       len(keypoints_one_person) != 17: # Ensure it's 17 keypoints
        return pd.Series(dtype='float64')

    try:
        data = {}
        for i, kp in enumerate(keypoints_one_person):
            data[f'y_{i}'] = kp[0] # Swapped x and y based on typical MoveNet output (y,x,s)
            data[f'x_{i}'] = kp[1]
            data[f's_{i}'] = kp[2]
        return pd.Series(data)
    except Exception:
        return pd.Series(dtype='float64')


def flatten_keypoints(keypoints_one_person): # Renamed
    # Expects a list of 17 keypoints, where each keypoint is [y, x, score]
    if isinstance(keypoints_one_person, list) and \
       len(keypoints_one_person) == 17 and \
       all(isinstance(kp, (list, tuple)) and len(kp) == 3 for kp in keypoints_one_person):
        return [v for kp in keypoints_one_person for v in kp] # Flattens to y0,x0,s0,y1,x1,s1,...
    return [] # Return empty list if input is not as expected


# Initialize session state
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
    st.session_state.svm_model = None
    st.session_state.label_encoder = None
    st.session_state.expected_features = EXPECTED_FEATURES


# File uploader for training
uploaded_files = st.file_uploader("Upload multiple boxing videos for Training/Processing", type=["mp4", "avi", "mov"], accept_multiple_files=True)

if uploaded_files:
    all_logs_for_training = [] # Changed variable name for clarity
    progress_bar = st.progress(0)
    
    for video_idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"📦 Processing for training data: {uploaded_file.name}")
        temp_dir_train = tempfile.mkdtemp() # Use a different temp_dir name
        input_path_train = os.path.join(temp_dir_train, uploaded_file.name)

        with open(input_path_train, 'wb') as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(input_path_train)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Note: OpenCV uses BGR by default. TF model might expect RGB.
        # Movenet from TF Hub usually handles NHWC uint8 [0,255] BGR or RGB.
        # For MultiPose Lightning, input is tf.int32, [1, height, width, 3]
        
        raw_output_train_path = os.path.join(temp_dir_train, "raw_output_train.mp4") # Different name
        out_writer = cv2.VideoWriter(raw_output_train_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        current_video_punch_log = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            st.warning(f"Could not get total frames for {uploaded_file.name}. Skipping.")
            continue
            
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Movenet expects input_size 256 for multipose lightning
            # Input tensor should be int32
            resized_frame = cv2.resize(frame, (256, 256)) 
            input_tensor = tf.cast(tf.convert_to_tensor(resized_frame[None, ...]), dtype=tf.int32)
            
            results = movenet.signatures['serving_default'](input_tensor)
            # keypoints_all_persons will be a list of lists. Each inner list is one person's 17 keypoints [y,x,s]
            keypoints_all_persons = extract_keypoints(results)


            if not keypoints_all_persons: # No one detected or confident enough
                out_writer.write(frame) # Write original frame
                frame_idx += 1
                if frame_idx % 10 == 0 : # Update progress less frequently
                    progress_percentage = min(((video_idx + (frame_idx / total_frames)) / len(uploaded_files)), 1.0)
                    progress_bar.progress(progress_percentage)
                continue

            # Get classifications for all detected people
            punches = classify_punch(keypoints_all_persons)
            postures = check_posture(keypoints_all_persons)
            gloves_statuses = detect_gloves(keypoints_all_persons)

            annotated_frame = draw_annotations(frame.copy(), keypoints_all_persons, punches, postures, gloves_statuses)
            out_writer.write(annotated_frame)

            for i, person_kps in enumerate(keypoints_all_persons): # Iterate through detected people
                current_video_punch_log.append({
                    "video": uploaded_file.name,
                    "frame": frame_idx,
                    "person_id_in_frame": i,
                    "punch": punches[i] if i < len(punches) else "N/A",
                    "posture": postures[i] if i < len(postures) else "N/A",
                    "gloves": gloves_statuses[i] if i < len(gloves_statuses) else "N/A",
                    "keypoints_for_person": person_kps # This is list of 17 [y,x,s] lists
                })
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                progress_percentage = min(((video_idx + (frame_idx / total_frames)) / len(uploaded_files)), 1.0)
                progress_bar.progress(progress_percentage)

        cap.release()
        out_writer.release()

        final_output_train_path = os.path.join(temp_dir_train, f"final_train_{uploaded_file.name}")
        try:
            ffmpeg.input(raw_output_train_path).output(final_output_train_path, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            st.video(final_output_train_path)
            with open(final_output_train_path, "rb") as f_video:
                st.download_button(f"📥 Download Annotated Training Video: {uploaded_file.name}", f_video, file_name=f"annotated_train_{uploaded_file.name}", mime="video/mp4")
        except ffmpeg.Error as e:
            st.error(f"FFmpeg error during training video processing: {e.stderr.decode('utf8') if e.stderr else 'Unknown ffmpeg error'}")
            st.info(f"Raw output was at: {raw_output_train_path}")


        df_video_log = pd.DataFrame(current_video_punch_log)
        if df_video_log.empty:
            st.warning(f"⚠️ No punch data logged for {uploaded_file.name}.")
        else:
            st.write(f"### 🔍 Keypoints Sample for {uploaded_file.name}")
            # Ensure 'keypoints_for_person' column exists before trying to access it
            if 'keypoints_for_person' in df_video_log.columns and not df_video_log['keypoints_for_person'].empty:
                 # Display first valid keypoint entry
                first_valid_keypoints = df_video_log['keypoints_for_person'].dropna().iloc[0] if not df_video_log['keypoints_for_person'].dropna().empty else "Not available"
                st.json(first_valid_keypoints)


            expanded_df_video = df_video_log.copy()
            # Apply expand_keypoints to the 'keypoints_for_person' column
            if 'keypoints_for_person' in expanded_df_video.columns:
                keypoint_features_df = expanded_df_video['keypoints_for_person'].apply(expand_keypoints)
                if not keypoint_features_df.empty: # Check if series is not empty
                    expanded_df_video = pd.concat([expanded_df_video.drop(columns=['keypoints_for_person']), keypoint_features_df], axis=1)
                else:
                     expanded_df_video = expanded_df_video.drop(columns=['keypoints_for_person']) # Drop if expansion failed

                st.dataframe(expanded_df_video.head())
                csv_data = expanded_df_video.to_csv(index=False).encode('utf-8')
                st.download_button(f"📄 Download Log CSV for {uploaded_file.name}", csv_data, file_name=f"log_{uploaded_file.name}.csv", mime="text/csv")
            all_logs_for_training.extend(current_video_punch_log) # Use extend with list of dicts

    progress_bar.empty()

    # Training models if log data is available
    if all_logs_for_training:
        df_all_logs = pd.DataFrame(all_logs_for_training)
        
        if 'keypoints_for_person' not in df_all_logs.columns or df_all_logs['keypoints_for_person'].isnull().all():
            st.error("Error: 'keypoints_for_person' data is missing or all null. Cannot train models.")
        else:
            df_all_logs["flat_kp"] = df_all_logs["keypoints_for_person"].apply(flatten_keypoints)
            # Filter out rows where flatten_keypoints returned an empty list (meaning invalid input)
            df_all_logs = df_all_logs[df_all_logs["flat_kp"].apply(lambda x: len(x) == EXPECTED_FEATURES)]

            if df_all_logs.empty:
                st.error("No valid keypoint data available for training after flattening. Please check video processing and keypoint extraction.")
            else:
                X = np.vstack(df_all_logs["flat_kp"].values)
                y = df_all_logs["punch"].values

                if len(np.unique(y)) < 2:
                    st.error(f"Not enough class diversity for training. Found only {len(np.unique(y))} unique punch types. Need at least 2.")
                else:
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)

                    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None)

                    svm_model_trained = svm.SVC(kernel='linear', probability=True) # Added probability for potential future use
                    svm_model_trained.fit(X_train, y_train)

                    tree_model_trained = DecisionTreeClassifier(max_depth=5, random_state=42)
                    tree_model_trained.fit(X_train, y_train)

                    svm_preds = svm_model_trained.predict(X_test)
                    tree_preds = tree_model_trained.predict(X_test)

                    st.write("### 📊 Model Evaluation on Training Data")
                    st.write(f"🔹 SVM Accuracy: {accuracy_score(y_test, svm_preds):.2f}")
                    st.write(classification_report(y_test, svm_preds, target_names=le.classes_, zero_division=0))
                    
                    st.write(f"🔹 Decision Tree Accuracy: {accuracy_score(y_test, tree_preds):.2f}")
                    st.write(classification_report(y_test, tree_preds, target_names=le.classes_, zero_division=0))


                    st.write("### Confusion Matrix (SVM)")
                    fig1, ax1 = plt.subplots()
                    ConfusionMatrixDisplay(confusion_matrix(y_test, svm_preds), display_labels=le.classes_).plot(ax=ax1, xticks_rotation='vertical')
                    st.pyplot(fig1)

                    st.write("### Confusion Matrix (Decision Tree)")
                    fig2, ax2 = plt.subplots()
                    ConfusionMatrixDisplay(confusion_matrix(y_test, tree_preds), display_labels=le.classes_).plot(ax=ax2, xticks_rotation='vertical')
                    st.pyplot(fig2)

                    # Save models to session state for the prediction part
                    st.session_state.svm_model = svm_model_trained
                    st.session_state.label_encoder = le
                    st.session_state.model_ready = True
                    st.session_state.expected_features = X_train.shape[1] 
                    st.success("✅ SVM and Decision Tree models trained and ready for prediction.")

                    # Option to save models to disk
                    # base_name_export = "boxing_analyzer_model" # Generic name
                    # dump(svm_model_trained, f"/tmp/{base_name_export}_svm_model.joblib")
                    # dump(tree_model_trained, f"/tmp/{base_name_export}_tree_model.joblib") # Unused in current prediction but good practice
                    # dump(le, f"/tmp/{base_name_export}_label_encoder.joblib")
                    # st.info("Models also saved to /tmp directory in the environment.")
    else:
        st.info("No training data was generated from the uploaded videos. Cannot train models.")


# Prediction Section
st.write("---")
st.header("🎬 Predict on a New Clip")

video_file_predict = st.file_uploader("Upload a test video for prediction", type=["mp4", "mov", "avi"], key="predict_uploader")

if video_file_predict is not None:
    if not st.session_state.model_ready or st.session_state.svm_model is None or st.session_state.label_encoder is None:
        st.error("🚫 Models are not yet trained or loaded. Please process training videos first in the section above.")
    else:
        # Load models from session state
        svm_model_predict = st.session_state.svm_model
        le_predict = st.session_state.label_encoder
        # EXPECTED_FEATURES is already global or can be fetched from st.session_state.expected_features

        st.info(f"👁️ Processing for prediction: {video_file_predict.name}")
        
        temp_dir_predict = tempfile.mkdtemp()
        input_path_predict = os.path.join(temp_dir_predict, "test_video.mp4") # Standardized name
        with open(input_path_predict, "wb") as f:
            f.write(video_file_predict.read())

        cap_predict = cv2.VideoCapture(input_path_predict)
        width_p = int(cap_predict.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_p = int(cap_predict.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_p = cap_predict.get(cv2.CAP_PROP_FPS)

        raw_output_predict_path = os.path.join(temp_dir_predict, "raw_output_predict.mp4")
        out_predict_writer = cv2.VideoWriter(raw_output_predict_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_p, (width_p, height_p))

        stframe_predict = st.empty()
        frame_count_predict = 0
        
        total_frames_predict = int(cap_predict.get(cv2.CAP_PROP_FRAME_COUNT))
        predict_progress = st.progress(0)

        while cap_predict.isOpened():
            ret, frame = cap_predict.read()
            if not ret:
                break
            
            current_frame_for_prediction = frame.copy() # Work on a copy

            try:
                resized_predict = cv2.resize(current_frame_for_prediction, (256, 256))
                input_tensor_predict = tf.cast(tf.convert_to_tensor(resized_predict[None, ...]), dtype=tf.int32)
                
                results_predict = movenet.signatures['serving_default'](input_tensor_predict)
                all_persons_kps_predict = extract_keypoints(results_predict)

                predicted_punches_texts = [] # For drawing

                if not all_persons_kps_predict:
                    cv2.putText(current_frame_for_prediction, "No person detected", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    # Draw skeletons first
                    for person_kps_item in all_persons_kps_predict:
                        # Draw skeleton for this person
                        for (p1_idx, p2_idx) in SKELETON_EDGES:
                            y1, x1, s1 = person_kps_item[p1_idx]
                            y2, x2, s2 = person_kps_item[p2_idx]
                            if s1 > 0.2 and s2 > 0.2:
                                pt1 = int(x1 * width_p), int(y1 * height_p) # Use prediction frame width/height
                                pt2 = int(x2 * width_p), int(y2 * height_p)
                                cv2.line(current_frame_for_prediction, pt1, pt2, (255, 100, 0), 2) # Blueish skeleton
                        # Draw keypoints
                        for (y, x, s) in person_kps_item:
                             if s > 0.2:
                                cx, cy = int(x * width_p), int(y * height_p)
                                cv2.circle(current_frame_for_prediction, (cx, cy), 3, (0, 255, 0), -1) # Green dots


                    for person_idx, person_kps in enumerate(all_persons_kps_predict):
                        if not person_kps or len(person_kps) != 17:
                            # This case should be rare if extract_keypoints is robust
                            st.warning(f"Frame {frame_count_predict}, P{person_idx}: Invalid keypoints structure.")
                            cv2.putText(current_frame_for_prediction, f"P{person_idx}: KPS Error", (30, 40 + person_idx * 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
                            continue

                        flat_kp_predict = flatten_keypoints(person_kps)

                        if len(flat_kp_predict) != st.session_state.expected_features:
                            st.warning(f"Frame {frame_count_predict}, P{person_idx}: Feature length mismatch. Got {len(flat_kp_predict)}, expected {st.session_state.expected_features}. Skipping prediction for this person.")
                            cv2.putText(current_frame_for_prediction, f"P{person_idx}: Feat Error (len {len(flat_kp_predict)})", (30, 40 + person_idx * 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            continue

                        X_input_predict = np.array(flat_kp_predict).reshape(1, -1)
                        pred_class_svm = svm_model_predict.predict(X_input_predict)
                        predicted_label_svm = le_predict.inverse_transform(pred_class_svm)[0]
                        
                        confidences_person = [kp[2] for kp in person_kps if isinstance(kp, (list, tuple)) and len(kp)==3]
                        avg_conf_person = np.mean(confidences_person) if confidences_person else 0.0
                        
                        # Position text near the person's head (nose keypoint: index 0)
                        nose_y, nose_x, nose_s = person_kps[0]
                        text_x_coord = int(nose_x * width_p) + 10 if nose_s > 0.2 else (30 + person_idx * 150)
                        text_y_start_coord = int(nose_y * height_p) - 20 if nose_s > 0.2 else 40

                        cv2.putText(current_frame_for_prediction, f"P{person_idx} Punch: {predicted_label_svm}", (text_x_coord, text_y_start_coord),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(current_frame_for_prediction, f"Conf: {avg_conf_person:.2f}", (text_x_coord, text_y_start_coord + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            except Exception as e:
                st.error(f"⚠️ Frame {frame_count_predict} prediction processing error: {str(e)}")
                # Optionally, put a general error message on the frame
                cv2.putText(current_frame_for_prediction, "Processing Error", (width_p // 2 - 100, height_p // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)


            out_predict_writer.write(current_frame_for_prediction)
            frame_rgb_predict = cv2.cvtColor(current_frame_for_prediction, cv2.COLOR_BGR2RGB)
            stframe_predict.image(frame_rgb_predict, caption=f"Frame {frame_count_predict + 1}", use_container_width=True)
            frame_count_predict += 1
            if total_frames_predict > 0 :
                 predict_progress.progress(frame_count_predict / total_frames_predict)


        cap_predict.release()
        out_predict_writer.release()
        predict_progress.empty()

        final_output_predict_path = os.path.join(temp_dir_predict, f"annotated_pred_{video_file_predict.name}")
        try:
            ffmpeg.input(raw_output_predict_path).output(final_output_predict_path, vcodec='libx264', acodec='aac', strict='experimental').run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            st.success(f"✅ Annotated prediction video ready: {video_file_predict.name}")
            st.video(final_output_predict_path)
            with open(final_output_predict_path, "rb") as f_pred_video:
                st.download_button(f"📥 Download Annotated Prediction Clip: {video_file_predict.name}", f_pred_video, file_name=f"annotated_pred_{video_file_predict.name}", mime="video/mp4")
        except ffmpeg.Error as e:
            st.error(f"FFmpeg error during prediction video encoding: {e.stderr.decode('utf8') if e.stderr else 'Unknown ffmpeg error'}")
            st.info(f"Raw prediction output was at: {raw_output_predict_path}")


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
''' # tqdm was listed but not directly used in the final script. Can be removed if not needed elsewhere.

with open("requirements.txt", "w") as f:
    f.write(requirements)
print("✅ requirements.txt saved")
