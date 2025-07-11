import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
import time
from datetime import datetime

@st.cache_resource
def load_models():
    clf = joblib.load("models/classifier.pkl")
    iso_model = joblib.load("models/anomaly_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return clf, iso_model, le

@st.cache_data
def load_data():
    X = np.load("data/X_scaled.npy")
    y = np.load("data/y.npy")
    return X, y

clf, iso_model, le = load_models()
X, y = load_data()
label_map = {i: label for i, label in enumerate(le.classes_)}

if "predictions" not in st.session_state:
    st.session_state.predictions = []

st.title("Real-Time Network Traffic Monitor")
st.subheader("AI-Based Classifier and Anomaly Detector")

placeholder = st.empty()

for i in range(50):
    idx = random.randint(0, len(X)-1)
    sample = X[idx].reshape(1, -1)
    true_label = y[idx]
    clf_pred = clf.predict(sample)[0]
    anomaly_pred = iso_model.predict(sample)[0]
    anomaly_pred = 0 if anomaly_pred == 1 else 1  # Convert -1 → 1

    clf_prob = clf.predict_proba(sample)[0]
    result = {
        "Index": idx,
        "True Label": label_map[true_label],
        "Classifier": label_map[clf_pred],
        "Confidence": round(clf_prob[clf_pred], 3),
        "Anomaly": label_map[anomaly_pred],
        "Time": datetime.now().strftime("%H:%M:%S")
    }

    st.session_state.predictions.append(result)
    df = pd.DataFrame(st.session_state.predictions)

    with placeholder.container():
        st.dataframe(df.tail(10), use_container_width=True)
        st.markdown("Prediction Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df["Classifier"].value_counts())
        with col2:
            st.bar_chart(df["Anomaly"].value_counts())

    time.sleep(1)
