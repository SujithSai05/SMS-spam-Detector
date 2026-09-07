import pickle
import os
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    return text


def train_model():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / 'data' / 'sms_data.csv'
    model_path = base_dir / 'model' / 'sms_spam_model.pkl'

    if not data_path.exists():
        raise FileNotFoundError(f'Dataset not found at {data_path}')

    df = pd.read_csv(data_path)
    df = df[['label', 'message']].dropna().reset_index(drop=True)
    df['message'] = df['message'].apply(clean_text)

    if set(df['label'].unique()) != {'ham', 'spam'}:
        df['label'] = df['label'].str.lower().str.strip()

    X = df['message']
    y = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True,
    )

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    model_dir = model_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'model': model}, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    train_model()
