# Dataset

Place your dataset here as `transcripts.csv`.

This project was built and tested using the **Twitter US Airline Sentiment**
dataset (real customer tweets about airlines, labeled Positive/Negative/Neutral):
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

Download `Tweets.csv` from Kaggle, place it here, and rename it to
`transcripts.csv`. Relevant columns used:
- `text` — the tweet content
- `airline_sentiment` — the sentiment label

Run training with:
```bash
py src/train.py --data Dataset/transcripts.csv --text-col text --label-col airline_sentiment
```
