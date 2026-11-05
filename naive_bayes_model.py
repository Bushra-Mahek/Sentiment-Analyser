import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from preprocessing import LABELS, LABEL_ORDER, clean_text

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "naive_bayes_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"


def create_model_directory() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_dataset() -> Dict[str, Any]:
    return load_dataset("cardiffnlp/tweet_eval", "sentiment")


def train_and_save_model() -> Dict[str, Any]:
    create_model_directory()

    dataset = load_training_dataset()

    train_texts = [clean_text(item["text"]) for item in dataset["train"]]
    train_labels = [int(item["label"]) for item in dataset["train"]]
    val_texts = [clean_text(item["text"]) for item in dataset["validation"]]
    val_labels = [int(item["label"]) for item in dataset["validation"]]
    test_texts = [clean_text(item["text"]) for item in dataset["test"]]
    test_labels = [int(item["label"]) for item in dataset["test"]]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    train_vectors = vectorizer.fit_transform(train_texts)
    naive_bayes_model = MultinomialNB(alpha=1.0, fit_prior=True)
    naive_bayes_model.fit(train_vectors, train_labels)

    val_predictions = naive_bayes_model.predict(vectorizer.transform(val_texts))
    test_predictions = naive_bayes_model.predict(vectorizer.transform(test_texts))

    validation_accuracy = float(np.mean(val_predictions == np.array(val_labels)))
    test_accuracy = float(np.mean(test_predictions == np.array(test_labels)))

    metadata = {
        "dataset_name": "cardiffnlp/tweet_eval",
        "dataset_variant": "sentiment",
        "train_samples": len(train_texts),
        "validation_samples": len(val_texts),
        "test_samples": len(test_texts),
        "class_names": {str(key): value for key, value in LABELS.items()},
        "sentiment_classes": ["Negative", "Neutral", "Positive"],
        "vectorizer_config": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
        },
        "naive_bayes_config": {
            "alpha": 1.0,
            "fit_prior": True,
        },
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
    }

    joblib.dump(naive_bayes_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    MODEL_INFO_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def load_model_bundle() -> Tuple[Any, Any]:
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        train_and_save_model()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def get_model_metadata() -> Dict[str, Any]:
    if MODEL_INFO_PATH.exists():
        return json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))

    return train_and_save_model()


def predict_sentiment(text: str) -> Dict[str, Any]:
    if text is None or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    model, vectorizer = load_model_bundle()
    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("Input text cannot be empty after preprocessing.")

    begin = time.perf_counter()
    transformed = vectorizer.transform([cleaned_text])
    probabilities = model.predict_proba(transformed)[0]
    prediction_index = int(model.predict(transformed)[0])
    prediction_label = LABELS.get(prediction_index, "Neutral")
    prediction_confidence = float(probabilities[prediction_index] * 100)

    distribution = {
        label: round(float(probabilities[index] * 100), 2)
        for index, label in LABELS.items()
    }

    return {
        "predicted_sentiment": prediction_label,
        "prediction_index": prediction_index,
        "probabilities": distribution,
        "confidence": round(prediction_confidence, 2),
        "inference_time_ms": round((time.perf_counter() - begin) * 1000, 2),
    }


def ensure_model_ready() -> Dict[str, Any]:
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists() and MODEL_INFO_PATH.exists():
        return get_model_metadata()
    return train_and_save_model()
