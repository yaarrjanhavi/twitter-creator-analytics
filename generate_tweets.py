import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed so you get same data each time (optional)
np.random.seed(42)
random.seed(42)

num_rows = 500

# 1) Create a range of timestamps (last 30 days, random times)
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

timestamps = [
    start_date + timedelta(
        days=random.uniform(0, 30),
        hours=random.uniform(0, 24)
    )
    for _ in range(num_rows)
]

# 2) Some simple fake tweet texts
sample_texts = [
    "Trying out a new AI tool",
    "Sharing some thoughts on growth and marketing",
    "Daily coding progress update",
    "New blog post is live now!",
    "Quick tip for data science beginners",
    "Building a side project this weekend",
    "Sharing my favorite productivity hack",
    "Here is a short thread on learning Python",
    "What do you think about this idea?",
    "Experimenting with different posting times"
]

# 3) Media types and hashtags
media_types = ["none", "image", "video", "link", "gif"]
hashtag_sets = [
    "",
    "growth,ai",
    "marketing,branding",
    "python,data",
    "productivity",
    "coding,learning",
    "startups,founder",
]

# 4) Generate synthetic engagement with some patterns:
#    - Evening (18–23) and media != none -> higher impressions & likes
tweet_ids = []
texts = []
impressions = []
likes = []
replies = []
retweets = []
profile_visits = []
media_type_col = []
hashtags_col = []
is_reply = []
is_thread = []

for i in range(num_rows):
    ts = timestamps[i]
    hour = ts.hour

    # basic tweet id
    tweet_ids.append(10_000_000 + i)

    # random text
    text = random.choice(sample_texts)
    texts.append(text)

    # random media and hashtags
    m_type = random.choice(media_types)
    media_type_col.append(m_type)
    htags = random.choice(hashtag_sets)
    hashtags_col.append(htags)

    # base impressions
    base_impressions = np.random.randint(100, 5000)

    # boost for evening + media
    if 18 <= hour <= 23 and m_type != "none":
        base_impressions = int(base_impressions * np.random.uniform(1.5, 3.0))

    # likes, replies, retweets derived from impressions (with noise)
    like_rate = np.random.uniform(0.01, 0.1)  # 1–10% of impressions
    reply_rate = np.random.uniform(0.001, 0.01)
    retweet_rate = np.random.uniform(0.002, 0.02)

    likes_val = int(base_impressions * like_rate)
    replies_val = int(base_impressions * reply_rate)
    retweets_val = int(base_impressions * retweet_rate)
    profile_visits_val = int(base_impressions * np.random.uniform(0.01, 0.2))

    impressions.append(base_impressions)
    likes.append(likes_val)
    replies.append(replies_val)
    retweets.append(retweets_val)
    profile_visits.append(profile_visits_val)

    # is_reply / is_thread basic pattern
    is_reply.append(np.random.choice([0, 1], p=[0.8, 0.2]))
    is_thread.append(np.random.choice([0, 1], p=[0.7, 0.3]))

# 5) Fake followers at post (slowly increasing over time)
#    Sort timestamps and assign more followers to later tweets
sorted_indices = np.argsort(timestamps)
followers = np.zeros(num_rows, dtype=int)
start_followers = 1000
end_followers = 5000

for rank, idx in enumerate(sorted_indices):
    # linear growth from 1000 to 5000 over 500 tweets
    frac = rank / (num_rows - 1)
    followers[idx] = int(start_followers + frac * (end_followers - start_followers))

# 6) Build DataFrame
df = pd.DataFrame({
    "tweet_id": tweet_ids,
    "timestamp": timestamps,
    "text": texts,
    "impressions": impressions,
    "likes": likes,
    "replies": replies,
    "retweets": retweets,
    "profile_visits": profile_visits,
    "followers_at_post": followers,
    "media_type": media_type_col,
    "hashtags": hashtags_col,
    "is_reply": is_reply,
    "is_thread": is_thread
})

# 7) Sort by timestamp (optional)
df = df.sort_values("timestamp").reset_index(drop=True)

# 8) Save to CSV
output_file = "tweets_synthetic.csv"
df.to_csv(output_file, index=False)

print(f"Saved {len(df)} synthetic tweets to {output_file}")
