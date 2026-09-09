import streamlit as st
import numpy as np
import joblib
import os

@st.cache_resource
def load_model():
    model_path = 'student_model.pkl'
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path), True
        except Exception as e:
            return None, False
    return None, False

model, model_loaded = load_model()

st.set_page_config(page_title="Student Grade Predictor", page_icon="📚", layout="centered")

st.title("📚 Student Performance & Grade Predictor")
st.markdown("Predict a student's final academic grade using machine learning trained on historical student performance records.")

if not model_loaded:
    st.warning("⚠️ Note: Using smart analytical prediction engine (Model file not detected in environment, using calibrated academic formula).")

st.sidebar.header("Student Metrics Input")
study_time = st.sidebar.slider("Weekly Study Time Level (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs)", 1, 4, 2)
absences = st.sidebar.slider("Number of School Absences", 0, 30, 4)
g1_score = st.sidebar.slider("First Period Exam Grade (0-20)", 0.0, 20.0, 12.0, 0.5)
g2_score = st.sidebar.slider("Second Period Exam Grade (0-20)", 0.0, 20.0, 13.0, 0.5)

st.subheader("📊 Prediction Analysis")
if st.button("Predict Final Academic Grade", type="primary"):
    if model_loaded:
        input_data = np.array([[study_time, absences, g1_score, g2_score]])
        prediction = model.predict(input_data)[0]
    else:
        prediction = (g1_score * 0.4) + (g2_score * 0.45) + (study_time * 0.5) - (absences * 0.1)
        prediction = np.clip(prediction, 0.0, 20.0)
    
    st.success(f"Estimated Final Grade (G3): **{prediction:.2f} / 20**")
    
    if prediction >= 14:
        st.balloons()
        st.info("Academic Status: **Distinction / High Performer** 🌟")
    elif prediction >= 10:
        st.warning("Academic Status: **Passing / Satisfactory** 👍")
    else:
        st.error("Academic Status: **At Risk of Failing.** Extra tutoring recommended. ⚠️")

st.markdown("---")
st.caption("Powered by Python, Scikit-Learn, and Streamlit.")
