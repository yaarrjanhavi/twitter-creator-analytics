# X/Twitter Creator Growth & Engagement Tool

Analyze your Twitter/X data, predict engagement for draft tweets, and discover your best posting times and content patterns.

## 🚀 Overview

This project is an end-to-end analytics tool for creators and marketers.  
You can:

- Load historical tweet data from CSV (or use a built-in synthetic dataset)
- See which hours, weekdays, and content types perform best
- Predict an engagement score for a new draft tweet before posting
- Explore simple dashboards built with Streamlit

## 🧱 Tech Stack

- Python
- pandas, numpy
- scikit-learn
- TextBlob
- Streamlit

## 📂 Project Structure
```
├── app/
│ ├── dashboard.py # Streamlit app
│ ├── data_utils.py # Data loading & cleaning
│ ├── features.py # Feature engineering
│ └── models.py # Model training & loading
├── data/
│ └── tweets_synthetic.csv # Demo synthetic dataset
├── models/
│ └── engagement_rf.pkl # Saved model (created after training)
└── generate_tweets.py # Script to generate synthetic tweets
```

## 🛠️ Setup

1. Clone the repository:

```
git clone https://github.com/yaarrjanhavi/twitter-creator-analytics.gi
cd twitter-creator-analytics
```

2. Install dependencies:
```
pip install pandas numpy scikit-learn streamlit textblob joblib
```


3. (Optional) Regenerate the synthetic dataset:
```
python generate_tweets.py
```


## ▶️ Run the App

From the project root:
```
cd app
streamlit run dashboard.py
```


Then open the URL shown in the terminal (usually `http://localhost:8501`).

## 💡 How to Use

- In the sidebar:
  - Choose **“Use built-in synthetic dataset”** or **“Upload my own CSV”**
- Explore:
  - Overview table and summary stats
  - Best hours and weekdays (bar charts)
- In the draft section:
  - Write a tweet, choose time, hashtags, and media type
  - Click **“Predict engagement score”** to compare different ideas

## 📜 License

This project is open-source and available under the MIT License.
