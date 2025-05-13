import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from sklearn import svm
from joblib import dump
import io

st.set_option('client.showErrorDetails', True)

st.title("🥊 Boxing Analyzer with Punches, Posture & Gloves")


