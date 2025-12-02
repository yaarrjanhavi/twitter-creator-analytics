import pandas as pd
from textblob import TextBlob

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add model features and engagement_score column.
    """
    df = df.copy()

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday  # 0=Monday

    # Text-based features
    df['text'] = df['text'].astype(str)
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().apply(len)

    # Hashtag features
    df['hashtags'] = df['hashtags'].fillna('')
    df['hashtag_count'] = df['hashtags'].apply(
        lambda x: 0 if x == '' else len(str(x).split(','))
    )

    # Media feature
    df['media_type'] = df['media_type'].fillna('none')
    df['has_media'] = (df['media_type'] != 'none').astype(int)

    # Simple sentiment feature
    def get_sentiment(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception:
            return 0.0

    df['sentiment'] = df['text'].apply(get_sentiment)

    # Engagement score: (likes + replies + retweets) / impressions
    for col in ['replies', 'retweets']:
        if col not in df.columns:
            df[col] = 0

    df['engagement_score'] = (
        (df['likes'] + df['replies'] + df['retweets']) / df['impressions']
    )

    # Handle any infinities or NaNs
    df['engagement_score'] = df['engagement_score'].replace([float('inf'), -float('inf')], 0)
    df['engagement_score'] = df['engagement_score'].fillna(0)

    return df

def get_feature_target(df: pd.DataFrame):
    """
    Split DataFrame into X (features) and y (target).
    """
    feature_cols = [
        'hour',
        'weekday',
        'text_length',
        'word_count',
        'hashtag_count',
        'has_media',
        'sentiment'
    ]

    X = df[feature_cols]
    y = df['engagement_score']
    return X, y, feature_cols
