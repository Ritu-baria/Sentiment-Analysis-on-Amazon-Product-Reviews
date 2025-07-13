# Sentiment-Analysis-on-Amazon-Product-Reviews# 🧠 Sentiment Analysis on Amazon Product Reviews

This project performs sentiment analysis on real customer reviews from the [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). It uses Natural Language Processing (NLP) techniques to classify reviews as **positive** or **negative** based on the review text and score.

---

## 📌 Project Overview

- 🎯 **Goal**: Predict sentiment (positive/negative) from Amazon review texts.
- 📄 **Dataset**: Includes review `Text` and `Score` (1 to 5 stars).
- ⚙️ **Method**:
  - Clean and preprocess text.
  - Label sentiment from review score.
  - Vectorize using TF-IDF.
  - Train Logistic Regression model.

---

## 📂 Dataset Info

- `Text`: The review content.
- `Score`: Integer rating (1 to 5).
- `Sentiment`: Derived from Score (positive ≥ 4, negative ≤ 3).

📌 Neutral reviews (score = 3) were excluded to make it a binary classification task.

---

## ⚙️ Tech Stack

- 🐍 Python
- 🧠 Scikit-learn (model training)
- 🧹 NLTK (text preprocessing)
- 📊 Matplotlib, Seaborn (visualization)
- 📄 Pandas, NumPy

---

## 🧪 Model Pipeline

1. Load and clean data (`Text`, `Score`)
2. Generate sentiment labels
3. Text preprocessing:
    - Lowercase, punctuation and stopword removal
4. TF-IDF vectorization
5. Logistic Regression model training
6. Model evaluation (classification report, confusion matrix)

---

## 🧾 How to Run

### ▶️ Step 1: Install Dependencies

```bash
pip install pandas scikit-learn nltk matplotlib seaborn
python -m nltk.downloader stopwords
