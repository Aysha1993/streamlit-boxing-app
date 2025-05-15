import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import tempfile
import os
import pandas as pd
import ffmpeg

st.set_page_config(layout="wide")
st.title("🥊 Boxing Analyzer with Pose, Punch, Glove & Posture Detection")


