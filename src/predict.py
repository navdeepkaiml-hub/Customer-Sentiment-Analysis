"""
Run sentiment inference on new customer transcripts using the trained model.

Usage:
    python src/predict.py --text "The support agent was extremely helpful and polite"
    python src/predict.py --input new_transcripts.csv --text-col transcript
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from preprocess import preprocess_series, preprocess_text

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_artifacts():
    model = joblib.load(MODELS_DIR / "sentiment_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer


def predict_text(text: str, model, vectorizer) -> str:
    cleaned = preprocess_text(text)
    X = vectorizer.transform([cleaned])
    return model.predict(X)[0]


def main():
    parser = argparse.ArgumentParser(description="Predict customer sentiment")
    parser.add_argument("--text", type=str, help="A single transcript string to classify")
    parser.add_argument("--input", type=str, help="CSV file with a text column to classify in bulk")
    parser.add_argument("--text-col", type=str, default="transcript", help="Text column name (for --input)")
    args = parser.parse_args()

    model, vectorizer = load_artifacts()

    if args.text:
        print(f"Sentiment: {predict_text(args.text, model, vectorizer)}")
    elif args.input:
        df = pd.read_csv(args.input)
        cleaned = preprocess_series(df[args.text_col])
        X = vectorizer.transform(cleaned)
        df["predicted_sentiment"] = model.predict(X)
        print(df[[args.text_col, "predicted_sentiment"]])
    else:
        parser.error("Provide either --text or --input")


if __name__ == "__main__":
    main()
