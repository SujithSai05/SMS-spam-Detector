# SMS Spam Detector

A complete NLP project built with Python to detect whether an SMS message is spam or legitimate (ham).

## Features
- Text preprocessing using NLP-friendly cleaning
- TF-IDF vectorization for text representation
- Logistic Regression classifier
- Model persistence with pickle
- Command-line prediction
- Simple Flask web interface

## Project Structure

- `data/sms_data.csv`: dataset used for training
- `src/train_model.py`: trains the SMS spam detection model
- `src/predict.py`: predicts the label for a given message
- `app.py`: small Flask web app for browser-based prediction
- `model/sms_spam_model.pkl`: saved trained model

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python src/train_model.py
   ```

4. Run prediction from the terminal:
   ```bash
   python src/predict.py "Win a free cash prize now!"
   ```

5. Run the web app:
   ```bash
   python app.py
   ```

   Then open: http://127.0.0.1:5000

## Example predictions
- "Hey, are we meeting at 6pm today?" -> Ham
- "Congratulations! You have won a free vacation." -> Spam

## Model details
The model uses:
- Lowercasing and text cleaning
- TF-IDF vectorization
- Logistic Regression classifier

This is a classic and effective baseline for text classification tasks.
