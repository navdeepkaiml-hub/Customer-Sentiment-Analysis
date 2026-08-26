"""
Customer Sentiment Analysis — Training Pipeline
=================================================
Loads customer transcripts (voice/chat/email), cleans and vectorizes the
text (TF-IDF), trains and compares Naive Bayes, Decision Tree, and KNN
classifiers, then saves the best model + vectorizer to disk.

Usage:
    python src/train.py --data Dataset/transcripts.csv --text-col transcript --label-col sentiment
"""

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from preprocess import preprocess_series

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
MODELS_DIR = ROOT / "models"


def load_data(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    return df


def run_eda(df: pd.DataFrame):
    IMAGES_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x="label", data=df, order=df["label"].value_counts().index)
    plt.title("Sentiment Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "sentiment_distribution.png")
    plt.close()

    text_length = df["clean_text"].astype(str).apply(len)
    plt.figure(figsize=(6, 4))
    sns.histplot(text_length, bins=40)
    plt.title("Transcript Length Distribution")
    plt.xlabel("Character count")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "text_length_distribution.png")
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    results = {}

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    results["Naive Bayes"] = (nb_model, y_pred_nb, accuracy_score(y_test, y_pred_nb))

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    results["Decision Tree"] = (dt_model, y_pred_dt, accuracy_score(y_test, y_pred_dt))

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    results["KNN"] = (knn_model, y_pred_knn, accuracy_score(y_test, y_pred_knn))

    return results


def evaluate(name, y_test, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (weighted):", precision_score(y_test, y_pred, average="weighted", zero_division=0))
    print("Recall (weighted):", recall_score(y_test, y_pred, average="weighted", zero_division=0))
    print("F1 Score (weighted):", f1_score(y_test, y_pred, average="weighted", zero_division=0))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))


def plot_confusion_matrix(name, y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(IMAGES_DIR / f"confusion_matrix_{safe_name}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train customer sentiment classification models")
    parser.add_argument("--data", type=str, default="Dataset/transcripts.csv", help="Path to CSV dataset")
    parser.add_argument("--text-col", type=str, default="transcript", help="Column containing raw text")
    parser.add_argument("--label-col", type=str, default="sentiment", help="Column containing sentiment label")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(args.data, args.text_col, args.label_col)

    print("Cleaning and preprocessing text...")
    df["clean_text"] = preprocess_series(df["text"])

    print("Running EDA (saving plots to images/)...")
    run_eda(df)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models...")
    results = train_models(X_train, X_test, y_train, y_test)

    labels = sorted(y.unique())
    for name, (model, y_pred, acc) in results.items():
        evaluate(name, y_test, y_pred)
        plot_confusion_matrix(name, y_test, y_pred, labels)

    best_name = max(results, key=lambda k: results[k][2])
    best_model = results[best_name][0]
    print(f"\nBest model: {best_name} ({results[best_name][2]:.4f} accuracy)")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "sentiment_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    print(f"Saved best model ({best_name}) to models/sentiment_model.pkl")


if __name__ == "__main__":
    main()
