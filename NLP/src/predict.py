import pickle
import sys
from pathlib import Path


def predict_message(message):
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / 'model' / 'sms_spam_model.pkl'

    if not model_path.exists():
        raise FileNotFoundError('Model not found. Please run: python src/train_model.py')

    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    vectorizer = saved['vectorizer']
    model = saved['model']

    processed = [message.lower().strip()]
    vectorized = vectorizer.transform(processed)
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    label = 'spam' if prediction == 1 else 'ham'
    confidence = max(probability)

    return label, confidence


if __name__ == '__main__':
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = input('Enter SMS message: ')

    label, confidence = predict_message(text)
    print(f'Prediction: {label.upper()}')
    print(f'Confidence: {confidence:.2%}')
