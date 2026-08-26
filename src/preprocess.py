"""
Text preprocessing utilities for customer sentiment analysis.

Cleans raw transcript text (voice/chat/email) before vectorization:
lowercasing, punctuation/number removal, stopword removal, and
tokenization/lemmatization.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers/extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Tokenize, drop stopwords, and lemmatize remaining tokens."""
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def preprocess_text(text: str) -> str:
    """Full pipeline: clean -> tokenize -> lemmatize."""
    return tokenize_and_lemmatize(clean_text(text))


def preprocess_series(series):
    """Apply preprocess_text to a pandas Series of raw transcripts."""
    return series.apply(preprocess_text)
