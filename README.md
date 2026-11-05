# Sentiment Analysis Comparative Studio

A simple Streamlit application that compares:

- Multinomial Naïve Bayes trained on the TweetEval sentiment dataset
- Google Gemini-based sentiment analysis using the same user input

## Features

- Naïve Bayes sentiment classification with actual three-class probabilities
- Gemini API sentiment scoring with structured JSON output
- Side-by-side comparison cards
- Recent session history
- CSV batch analysis support
- Automatic model persistence using Joblib

## Setup

1. Create a Python environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Optionally create a `.env` file using `.env.example`.
4. Launch the application:

   streamlit run app.py

## Environment variables

- `GEMINI_API_KEY` - optional API key for Gemini analysis
- `GEMINI_MODEL` - optional model name, default `gemini-2.0-flash`

## Notes

- Naïve Bayes works without a Gemini key.
- Gemini analysis requires a valid API key and internet access.
- The training pipeline downloads the public `cardiffnlp/tweet_eval` dataset.
