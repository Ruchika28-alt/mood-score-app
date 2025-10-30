
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title='Mood Score Predictor')

st.title('Mood Score Predictor')
st.markdown('Enter your digital habits to predict mood_score using trained models.')

@st.cache_resource
def load_models():
    lr = joblib.load('/content/saved_models/linear_regression_pipeline.joblib')
    rf = joblib.load('/content/saved_models/random_forest_pipeline.joblib')
    return lr, rf

lr_model, rf_model = load_models()

features = ['screen_time_hours', 'social_media_platforms_used', 'hours_on_TikTok', 'sleep_hours', 'stress_level']

inputs = {}
for f in features:
    inputs[f] = st.number_input(f, value=0.0)

if st.button('Predict mood_score'):
    X_input = pd.DataFrame([inputs])
    pred_lr = lr_model.predict(X_input)[0]
    pred_rf = rf_model.predict(X_input)[0]
    st.write(f'**Linear Regression Prediction:** {pred_lr:.2f}')
    st.write(f'**Random Forest Prediction:** {pred_rf:.2f}')
    st.write(f'**Average Prediction:** {(pred_lr + pred_rf)/2:.2f}')
