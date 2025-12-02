import pandas as pd

def load_tweets_csv(path: str) -> pd.DataFrame:
    """
    Load and lightly clean a tweets CSV file.
    """
    df = pd.read_csv(path)

    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Basic cleaning
    if 'tweet_id' in df.columns:
        df = df.drop_duplicates(subset=['tweet_id'])

    # Drop rows with missing critical fields
    for col in ['text', 'impressions', 'likes']:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # Keep only rows with positive impressions
    if 'impressions' in df.columns:
        df = df[df['impressions'] > 0]

    # Fill missing optional columns if needed
    for col in ['media_type', 'hashtags', 'replies', 'retweets',
                'profile_visits', 'followers_at_post', 'is_reply', 'is_thread']:
        if col not in df.columns:
            if col in ['replies', 'retweets', 'profile_visits', 'followers_at_post',
                       'is_reply', 'is_thread']:
                df[col] = 0
            else:
                df[col] = ""

    return df
