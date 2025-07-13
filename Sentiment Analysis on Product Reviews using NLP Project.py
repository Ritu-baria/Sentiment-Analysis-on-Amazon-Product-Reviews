# Sentiment Analysis on Product Reviews using NLP Project.py
# Untitled-1
# Step 1: Import libraries
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')

# Step 2: Load dataset
df = pd.read_csv("Reviews.csv")  # Replace with your actual CSV filename

# Step 3: Select and clean necessary columns
df = df[['Score', 'Text']].dropna()

# Step 4: Create binary sentiment labels
df = df[df['Score'] != 3]  # Remove neutral reviews (optional)
df['sentiment'] = df['Score'].apply(lambda x: 'positive' if x >= 4 else 'negative')

# Step 5: Clean the review text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return " ".join(filtered)

df['clean_text'] = df['Text'].apply(clean_text)

# Step 6: Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 9: Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Step 10: Confusion matrix plot
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
