import os
import datetime as dt

import pandas as pd
import streamlit as st
from textblob import TextBlob

from data_utils import load_tweets_csv
from features import add_features, get_feature_target
from models import train_engagement_model, load_engagement_model



MODEL_PATH = "models/engagement_rf.pkl"
DEFAULT_CSV_PATH = "../data/tweets_synthetic.csv"


st.set_page_config(page_title="X/Twitter Creator Analytics", layout="wide")

st.title("X/Twitter Creator Growth & Engagement Tool")

st.sidebar.header("1. Load your data")

data_source = st.sidebar.radio(
    "Choose data source:",
    ("Use built-in synthetic dataset", "Upload my own CSV")
)

df = None

if data_source == "Use built-in synthetic dataset":
    if os.path.exists(DEFAULT_CSV_PATH):
        df = load_tweets_csv(DEFAULT_CSV_PATH)
        st.sidebar.success(f"Loaded synthetic dataset: {DEFAULT_CSV_PATH}")
    else:
        st.sidebar.error(
            f"Could not find {DEFAULT_CSV_PATH}. Make sure you ran the generator script."
        )
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = load_tweets_csv(uploaded_file)
        st.sidebar.success("Uploaded and loaded your CSV file.")

if df is None:
    st.info("Load a dataset from the sidebar to get started.")
    st.stop()

# ====== DATA OVERVIEW TAB ======
st.subheader("Data overview")
st.write("First 10 rows of your data:")
st.dataframe(df.head(10))

st.markdown("---")

# ====== FEATURE ENGINEERING ======
st.subheader("Feature engineering and summary")

df_feat = add_features(df)
st.write("Basic statistics for key metrics:")
st.write(df_feat[['impressions', 'likes', 'replies', 'retweets',
                  'profile_visits', 'engagement_score']].describe())

# Best hour and weekday plots
st.markdown("### Best time to post (by average engagement score)")

col1, col2 = st.columns(2)

with col1:
    st.caption("Average engagement by hour of day")
    best_hour = df_feat.groupby('hour')['engagement_score'].mean().sort_index()
    st.bar_chart(best_hour)

with col2:
    st.caption("Average engagement by weekday (0=Mon, 6=Sun)")
    best_weekday = df_feat.groupby('weekday')['engagement_score'].mean().sort_index()
    st.bar_chart(best_weekday)

# Simple "insight" text
overall_avg = df_feat['engagement_score'].mean()
top_hours = best_hour.sort_values(ascending=False).head(3).index.tolist()
top_weekdays = best_weekday.sort_values(ascending=False).head(2).index.tolist()

st.markdown("### Simple insights")
st.write(f"- Overall average engagement score: **{overall_avg:.4f}**")
st.write(f"- Top 3 hours (by engagement): **{top_hours}**")
st.write(f"- Top 2 weekdays (by engagement): **{top_weekdays}**")

st.markdown("---")

# ====== MODEL TRAINING / LOADING ======
st.subheader("Model training")

if os.path.exists(MODEL_PATH):
    model = load_engagement_model(MODEL_PATH)
    st.success("Loaded existing model from disk.")
    retrain = st.checkbox("Retrain model on current data", value=False)
else:
    st.info("No existing model found. Need to train a new one.")
    retrain = True

if retrain:
    X, y, feature_cols = get_feature_target(df_feat)
    with st.spinner("Training model..."):
        model, mae, r2 = train_engagement_model(X, y, MODEL_PATH)
    st.success(f"Trained model. MAE = {mae:.5f}, R² = {r2:.3f}")
else:
    # Quick evaluation on current data if user wants
    if st.checkbox("Evaluate loaded model on current data"):
        X, y, feature_cols = get_feature_target(df_feat)
        from sklearn.metrics import mean_absolute_error, r2_score
        y_pred_all = model.predict(X)
        mae_all = mean_absolute_error(y, y_pred_all)
        r2_all = r2_score(y, y_pred_all)
        st.write(f"Current data - MAE: {mae_all:.5f}, R²: {r2_all:.3f}")

st.markdown("---")

# ====== DRAFT TWEET PREDICTION ======
st.subheader("Predict engagement for a draft tweet")

col_left, col_right = st.columns([2, 1])

with col_left:
    draft_text = st.text_area(
        "Write your tweet:",
        value="Trying out a new AI tool for Twitter analytics 🚀"
    )

    hashtags_input = st.text_input(
        "Hashtags (comma-separated, optional)",
        value="ai,growth"
    )

with col_right:
    post_date = st.date_input("Planned date", dt.date.today())
    post_time = st.time_input("Planned time", dt.time(21, 0))  # default 9pm
    media_option = st.selectbox(
        "Media type",
        ["none", "image", "video", "link", "gif"],
        index=1
    )

if st.button("Predict engagement score"):
    ts = dt.datetime.combine(post_date, post_time)
    hour = ts.hour
    weekday = ts.weekday()
    text_length = len(draft_text)
    word_count = len(draft_text.split())

    if hashtags_input.strip() == "":
        hashtag_count = 0
    else:
        hashtag_count = len(hashtags_input.split(","))

    has_media = 0 if media_option == "none" else 1
    sentiment = TextBlob(draft_text).sentiment.polarity

    X_new = pd.DataFrame([{
        "hour": hour,
        "weekday": weekday,
        "text_length": text_length,
        "word_count": word_count,
        "hashtag_count": hashtag_count,
        "has_media": has_media,
        "sentiment": sentiment
    }])

    pred_score = model.predict(X_new)[0]
    st.success(f"Predicted engagement score: **{pred_score:.5f}**")
    st.info(
        "Higher is better. Try changing time, media, or text and compare the score."
    )
