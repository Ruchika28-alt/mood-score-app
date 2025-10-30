import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# PAGE CONFIG & STYLING
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
    h1, h2, h3, p, label {
        color: #2c3e50;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #ffffffcc;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
        text-align: center;
        margin-top: 1rem;
    }
    .stNumberInput label {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# TITLE & DESCRIPTION
# -------------------------------
st.title("💫 Mood Score Predictor")
st.markdown(
    """
    ### 🌿 Predict your mood based on your digital habits  
    Enter your daily screen time, social media usage, and sleep patterns.  
    The app will estimate your **mood level** using trained ML models.  
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
        st.error("❌ Model files not found! Please upload them in the `saved_models` folder.")
        st.stop()

    lr = joblib.load(lr_path)
    rf = joblib.load(rf_path)
    return lr, rf

lr_model, rf_model = load_models()

# -------------------------------
# INPUT SECTION
# -------------------------------
st.markdown("## 📋 Enter Your Daily Digital Habits")

col1, col2 = st.columns(2)
with col1:
    screen_time_hours = st.number_input("📱 Screen Time (hours/day)", min_value=0.0, step=0.1)
    social_media_platforms_used = st.number_input("🌐 Social Media Platforms Used", min_value=0, step=1)
    hours_on_TikTok = st.number_input("🎵 Hours on TikTok", min_value=0.0, step=0.1)
with col2:
    sleep_hours = st.number_input("💤 Sleep Hours", min_value=0.0, step=0.1)
    stress_level = st.number_input("⚡ Stress Level (1–10)", min_value=1, max_value=10, step=1)

inputs = {
    "screen_time_hours": screen_time_hours,
    "social_media_platforms_used": social_media_platforms_used,
    "hours_on_TikTok": hours_on_TikTok,
    "sleep_hours": sleep_hours,
    "stress_level": stress_level
}

# -------------------------------
# PREDICTION SECTION
# -------------------------------
if st.button("✨ Predict My Mood"):
    X_input = pd.DataFrame([inputs])

    pred_lr = lr_model.predict(X_input)[0]
    pred_rf = rf_model.predict(X_input)[0]
    avg_pred = (pred_lr + pred_rf) / 2

    # Clip predictions to valid range
    pred_lr = np.clip(pred_lr, 1, 10)
    pred_rf = np.clip(pred_rf, 1, 10)
    avg_pred = np.clip(avg_pred, 1, 10)

    # Mood interpretation function
    def interpret_mood(score):
        if score <= 3:
            return ("😞 Very Low Mood", "#e74c3c")
        elif score <= 5:
            return ("😐 Low Mood", "#e67e22")
        elif score <= 7:
            return ("🙂 Moderate Mood", "#f1c40f")
        elif score <= 8.5:
            return ("😄 Good Mood", "#2ecc71")
        else:
            return ("🤩 Excellent Mood", "#27ae60")

    mood_lr, color_lr = interpret_mood(pred_lr)
    mood_rf, color_rf = interpret_mood(pred_rf)
    mood_avg, color_avg = interpret_mood(avg_pred)

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            f"""
            <div class="prediction-box">
            <h3>🧮 Linear Regression</h3>
            <h2 style='color:{color_lr};'>{mood_lr}</h2>
            <p><i>Predicted Score: {pred_lr:.2f}</i></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="prediction-box">
            <h3>🌲 Random Forest</h3>
            <h2 style='color:{color_rf};'>{mood_rf}</h2>
            <p><i>Predicted Score: {pred_rf:.2f}</i></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="prediction-box">
        <h3>💡 Final Mood Prediction</h3>
        <h1 style='color:{color_avg};'>{mood_avg}</h1>
        <p><i>Average Score: {avg_pred:.2f}</i></p>
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
