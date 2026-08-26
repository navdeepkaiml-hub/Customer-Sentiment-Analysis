<div align="center">

# 💬 Customer Sentiment Analysis

### Classifying customer sentiment from text using NLP + Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-2E8B57?style=for-the-badge)](https://www.nltk.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

![Last Commit](https://img.shields.io/github/last-commit/navdeepkaiml-hub/Customer-Sentiment-Analysis)
![Repo Size](https://img.shields.io/github/repo-size/navdeepkaiml-hub/Customer-Sentiment-Analysis)

</div>

---

## 📖 Overview

Companies collect massive volumes of customer feedback across **social media, chat, and email** — but reading every message manually doesn't scale. This project uses **Natural Language Processing (NLP)** and classical **Machine Learning** to automatically classify customer sentiment as **Positive**, **Negative**, or **Neutral**, using real-world airline customer tweets as the dataset.

🏆 **Best model:** Naive Bayes — **75.3% accuracy**

---

## 🖼️ Results Preview

<div align="center">
<img src="images/sentiment_distribution.png" width="45%" alt="Sentiment Class Distribution"/>
<img src="images/text_length_distribution.png" width="45%" alt="Transcript Length Distribution"/>
</div>

<div align="center">
<img src="images/confusion_matrix_naive_bayes.png" width="45%" alt="Naive Bayes Confusion Matrix"/>
<img src="images/confusion_matrix_decision_tree.png" width="45%" alt="Decision Tree Confusion Matrix"/>
</div>

---

## 🗂️ Project Structure

```
Customer-Sentiment-Analysis/
├── Dataset/                # Raw transcript data
├── notebooks/
│   └── Code.ipynb           # EDA + modeling notebook
├── src/
│   ├── preprocess.py         # Text cleaning, tokenization, lemmatization
│   ├── train.py              # Training pipeline (CLI)
│   └── predict.py            # Inference on new transcripts
├── images/                  # Saved plots
├── models/                  # Trained model artifacts (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🎯 Objectives

- 🔍 Analyze customer transcripts using NLP techniques
- 🏷️ Classify sentiment as Positive / Negative / Neutral
- 📊 Identify key insights and patterns via EDA
- 🆚 Compare multiple ML models on real-world text data
- 💡 Provide a reusable, reproducible pipeline

---

## ⚙️ Methodology

| Step | Description |
|------|-------------|
| 1️⃣ Data Collection | Real-world airline customer tweets dataset |
| 2️⃣ Text Preprocessing | Lowercasing, punctuation removal, stopword removal, lemmatization |
| 3️⃣ Feature Extraction | TF-IDF vectorization (unigrams + bigrams) |
| 4️⃣ EDA | Class distribution, transcript length distribution |
| 5️⃣ Modeling | Naive Bayes, Decision Tree, KNN |
| 6️⃣ Evaluation | Accuracy, Precision, Recall, F1-score, Confusion Matrix |

---

## 📊 Results

| Model | Accuracy |
|-------|:--------:|
| 🥇 **Naive Bayes** | **75.3%** |
| 🥈 KNN | 69.7% |
| 🥉 Decision Tree | 66.9% |

**Key Insights:**
- 📉 Negative sentiment is the majority class and is predicted most reliably across all models
- 😐 Neutral sentiment is the hardest class to detect — likely due to ambiguous/short text
- 🥇 Naive Bayes generalizes best on short, sparse text like tweets

---

## 🚀 Getting Started

### 1️⃣ Clone the repo
```bash
git clone https://github.com/navdeepkaiml-hub/Customer-Sentiment-Analysis.git
cd Customer-Sentiment-Analysis
```

### 2️⃣ Install dependencies
```bash
py -m pip install -r requirements.txt
```

### 3️⃣ Train the models
```bash
py src/train.py --data Dataset/transcripts.csv --text-col text --label-col airline_sentiment
```

### 4️⃣ Predict on new text
```bash
py src/predict.py --text "The support agent resolved my issue quickly and was very polite"
```

### 📓 Or explore interactively
```bash
jupyter notebook notebooks/Code.ipynb
```

---

## 🛠️ Tech Stack

<div align="left">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/-NLTK-2E8B57?style=flat-square)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?style=flat-square)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

---

## 🔭 Future Work

- 🤖 Try deep learning (LSTM, BERT) for higher accuracy
- 🎯 Extend to aspect-based sentiment analysis
- 😊 Add emotion detection beyond positive/negative/neutral
- 🌐 Deploy as a real-time API for live text scoring

---

## 👤 Author

**Navdeep Kumar**
B.Tech CSE (AI & ML) — Roorkee Institute of Technology

[![GitHub](https://img.shields.io/badge/GitHub-navdeepkaiml--hub-181717?style=flat-square&logo=github)](https://github.com/navdeepkaiml-hub)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
