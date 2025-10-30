import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# PAGE CONFIG & CUSTOM STYLING
# -------------------------------
st.set_page_config(
    page_title="Mood Score Predictor",
    page_icon="💫",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        font-family: "Poppins", sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
        color: #2c3e50;
    }
    .css-1d391kg, .css-12ttj6m {
        background-color: rgba(255, 255, 255, 0.85) !important;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #ffffffcc;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
        text-align: center;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# TITLE & INTRO
# -------------------------------
st.title("💫 Mood Score Predictor")
st.markdown(
    """
    ### 🌿 Predict your mood based on your digital habits  
    Fill in your average daily screen time, social media use, and sleep patterns.  
    The app will estimate your **mood score** using two trained ML models.  
    """
)

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)
    lr_path = os.path.join(base_dir, "saved_models", "linear_regression_pipeline.joblib")
    rf_path = os.path.join(base_dir, "saved_models", "random_forest_pipeline.joblib")

    if not os.path.exists(lr_path) or not os.path.exists(rf_path):
        st.error("❌ Model files not found! Make sure they are in the `saved_models` folder.")
        st.stop()

    lr = joblib.load(lr_path)
    rf = joblib.load(rf_path)
    return lr, rf

lr_model, rf_model = load_models()

# -------------------------------
# USER INPUTS
# -------------------------------
st.markdown("## 📋 Enter Your Daily Digital Habits")

col1, col2 = st.columns(2)
with col1:
    screen_time_hours = st.number_input("📱 Screen Time (hours/day)", min_value=0.0, step=0.1)
    social_media_platforms_used = st.number_input("🌐 Social Media Platforms Used", min_value=0, step=1)
    hours_on_TikTok = st.number_input("🎵 Hours on TikTok", min_value=0.0, step=0.1)
with col2:
    sleep_hours = st.number_input("💤 Sleep Hours", min_value=0.0, step=0.1)
    stress_level = st.number_input("⚡ Stress Level (1-10)", min_value=1, max_value=10, step=1)

inputs = {
    "screen_time_hours": screen_time_hours,
    "social_media_platforms_used": social_media_platforms_used,
    "hours_on_TikTok": hours_on_TikTok,
    "sleep_hours": sleep_hours,
    "stress_level": stress_level
}

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("✨ Predict My Mood Score"):
    X_input = pd.DataFrame([inputs])

    pred_lr = lr_model.predict(X_input)[0]
    pred_rf = rf_model.predict(X_input)[0]
    avg_pred = (pred_lr + pred_rf) / 2

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            f"""
            <div class="prediction-box">
            <h3>🧮 Linear Regression</h3>
            <h2 style='color:#2980b9;'>{pred_lr:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="prediction-box">
            <h3>🌲 Random Forest</h3>
            <h2 style='color:#27ae60;'>{pred_rf:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="prediction-box">
        <h3>💡 Final Average Prediction</h3>
        <h1 style='color:#8e44ad;'>{avg_pred:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown(
    """
    ---
    Made with ❤️ using [Streamlit](https://streamlit.io)  
    [GitHub Repo](https://github.com/YOUR_USERNAME/mood-score-app)
    """,
    unsafe_allow_html=True
)
