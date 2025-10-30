import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="Mood Score Predictor", page_icon="💫")

st.title("💫 Mood Score Predictor")
st.markdown(
    """
    ### 🌿 Predict your mood based on your digital habits  
    Enter your daily screen time, social media use, and sleep patterns.  
    The app will estimate your **mood level** using trained ML models.  
    """
)

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    lr_path = os.path.join("saved_models", "linear_regression_pipeline.joblib")
    rf_path = os.path.join("saved_models", "random_forest_pipeline.joblib")

    if not os.path.exists(lr_path) or not os.path.exists(rf_path):
        st.error("❌ Model files not found! Make sure both are in `saved_models/`.")
        st.stop()

    lr = joblib.load(lr_path)
    rf = joblib.load(rf_path)
    return lr, rf

lr_model, rf_model = load_models()

# -------------------------------
# INPUT FORM
# -------------------------------
st.header("📋 Enter Your Daily Digital Habits")

features = [
    "screen_time_hours",
    "social_media_platforms_used",
    "hours_on_TikTok",
    "sleep_hours",
    "stress_level",
]

col1, col2 = st.columns(2)
inputs = {}

with col1:
    inputs["screen_time_hours"] = st.number_input("📱 Screen Time (hours/day)", min_value=0.0, step=0.1)
    inputs["social_media_platforms_used"] = st.number_input("🌐 Social Media Platforms Used", min_value=0, step=1)
    inputs["hours_on_TikTok"] = st.number_input("🎵 Hours on TikTok", min_value=0.0, step=0.1)
with col2:
    inputs["sleep_hours"] = st.number_input("💤 Sleep Hours", min_value=0.0, step=0.1)
    inputs["stress_level"] = st.number_input("⚡ Stress Level (1–10)", min_value=1, max_value=10, step=1)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("✨ Predict Mood"):
    X_input = pd.DataFrame([inputs])

    pred_lr = lr_model.predict(X_input)[0]
    pred_rf = rf_model.predict(X_input)[0]
    avg_pred = (pred_lr + pred_rf) / 2

    # Clip predictions to valid 1–10 range
    pred_lr = np.clip(pred_lr, 1, 10)
    pred_rf = np.clip(pred_rf, 1, 10)
    avg_pred = np.clip(avg_pred, 1, 10)

    # Interpret mood text + color
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

    st.divider()
    st.subheader("🧠 Predictions")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            f"<h4>🧮 Linear Regression</h4><h3 style='color:{color_lr};'>{mood_lr}</h3><p><i>Score: {pred_lr:.2f}</i></p>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<h4>🌲 Random Forest</h4><h3 style='color:{color_rf};'>{mood_rf}</h3><p><i>Score: {pred_rf:.2f}</i></p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style='text-align:center; margin-top:20px; padding:15px; border-radius:10px;
                    background-color:#f9f9f9; box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
            <h3>💡 Final Mood Prediction</h3>
            <h2 style='color:{color_avg};'>{mood_avg}</h2>
            <p><i>Average Score: {avg_pred:.2f}</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | [GitHub Repo](https://github.com/YOUR_USERNAME/mood-score-app)")
