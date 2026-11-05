import os
from pathlib import Path

import pandas as pd
import streamlit as st

from comparison import compare_predictions
from llm_analyzer import analyze_text
from naive_bayes_model import ensure_model_ready, get_model_metadata, predict_sentiment

st.set_page_config(page_title="Sentiment Analysis Comparative Studio", layout="wide")

SAMPLE_REVIEWS = {
    "Positive": "I absolutely love this product. The quality is excellent and it works perfectly.",
    "Negative": "This is a terrible product. It stopped working after two days and I regret buying it.",
    "Neutral": "The product arrived yesterday. It has the features mentioned in the description.",
    "Sarcastic": "Oh fantastic, another update that completely broke the application. Exactly what I needed.",
    "Mixed": "The camera quality is excellent, but the battery life is disappointing.",
}


def render_probability_bars(probabilities: dict, label_prefix: str = "") -> None:
    for label in ["Positive", "Negative", "Neutral"]:
        value = float(probabilities.get(label, 0.0))
        st.markdown(f"{label_prefix}{label}: {value:.2f}%")
        st.progress(min(max(value / 100.0, 0.0), 1.0))


def initialize_session_state() -> None:
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""


initialize_session_state()
ensure_model_ready()

st.title("Sentiment Analysis Comparative Studio")
st.caption("Naïve Bayes vs LLM-Based Sentiment Analysis")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API key", type="password", help="Paste your Gemini API key here. It stays in memory only.")
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

    st.caption("Naïve Bayes works without a Gemini API key. Gemini analysis requires a valid key.")
    st.caption("The key is never displayed or stored in the source code.")

    model_name = st.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"), help="Use a supported Gemini model name.")

    st.markdown("---")
    metadata = get_model_metadata()
    st.subheader("Model information")
    st.write(f"Dataset: {metadata.get('dataset_name', 'cardiffnlp/tweet_eval')}")
    st.write(f"Training samples: {metadata.get('train_samples', 'N/A')}")
    st.write(f"Sentiment classes: {', '.join(metadata.get('sentiment_classes', ['Negative', 'Neutral', 'Positive']))}")
    st.write(f"Vectorizer: {metadata.get('vectorizer_config', {})}")
    st.write(f"Naïve Bayes: {metadata.get('naive_bayes_config', {})}")
    st.write(f"LLM model: {model_name or 'gemini-3.5-flash-lite'}")

st.subheader("Input review")
input_text = st.text_area(
    "Enter text to analyse",
    value=st.session_state.text_input,
    height=220,
    help="Type or paste any sentence, statement, or review for sentiment analysis.",
)

st.session_state.text_input = input_text

char_count = len(input_text)
col_input, col_clear = st.columns([3, 1])
with col_input:
    st.caption(f"Character count: {char_count}")
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.text_input = ""
        st.rerun()

sample_cols = st.columns(len(SAMPLE_REVIEWS))
for idx, (label, sample_text) in enumerate(SAMPLE_REVIEWS.items()):
    with sample_cols[idx]:
        if st.button(label, key=f"sample_{label}", use_container_width=True):
            st.session_state.text_input = sample_text
            st.rerun()

analyse_clicked = st.button("Analyse", type="primary")

if analyse_clicked:
    if not input_text.strip():
        st.warning("Please enter a sentence before running the analysis.")
    else:
        with st.spinner("Running both sentiment models..."):
            nb_result = predict_sentiment(input_text)
            llm_result = analyze_text(input_text, api_key=api_key, model_name=model_name)
            comparison = compare_predictions(nb_result, llm_result)

        st.session_state.analysis_history.insert(
            0,
            {
                "input_text": input_text,
                "nb_prediction": nb_result["predicted_sentiment"],
                "llm_prediction": llm_result.get("sentiment", "Unavailable"),
                "agreement": "Agree" if comparison["agree"] else "Disagree",
            },
        )
        st.session_state.analysis_history = st.session_state.analysis_history[:10]

        nb_col, llm_col = st.columns(2)

        with nb_col:
            st.markdown("### Naïve Bayes")
            st.markdown(f"Predicted sentiment: **{nb_result['predicted_sentiment']}**")
            render_probability_bars(nb_result["probabilities"])
            st.caption(f"Actual prediction confidence: {nb_result['confidence']:.2f}%")
            st.caption(f"Inference time: {nb_result['inference_time_ms']} ms")

        with llm_col:
            st.markdown("### Gemini LLM")
            if llm_result.get("error"):
                st.warning(llm_result["error"])
                st.markdown("Predicted sentiment: **Unavailable**")
                st.caption("LLM-estimated scores are not available because the request failed.")
            else:
                st.markdown(f"Predicted sentiment: **{llm_result['sentiment']}**")
                render_probability_bars(
                    {
                        "Positive": llm_result["scores"].get("positive", 0.0),
                        "Negative": llm_result["scores"].get("negative", 0.0),
                        "Neutral": llm_result["scores"].get("neutral", 0.0),
                    },
                    label_prefix="LLM-estimated ",
                )
                st.caption("These values are LLM-estimated sentiment scores, not statistically calibrated probabilities.")
                st.caption(f"Explanation: {llm_result['explanation']}")
                st.caption(f"Emotional tone: {llm_result['emotional_tone']}")
                st.caption(f"Negation detected: {llm_result['contains_negation']}")
                st.caption(f"Sarcasm detected: {llm_result['contains_sarcasm']}")
                st.caption(f"API response latency: {llm_result['latency_ms']} ms")

        st.markdown("---")
        st.subheader("Model comparison")
        if comparison["agree"]:
            st.success(f"Result: {comparison['summary']}")
        else:
            st.warning(f"Result: {comparison['summary']}")

        st.write(f"Naïve Bayes: {comparison['nb_prediction']}")
        st.write(f"Gemini: {comparison['llm_prediction']}")
        st.write(f"Difference in sentiment interpretation: {comparison['difference']}")
        st.write(comparison["explanation"])

st.markdown("---")
st.subheader("Analysis history")
if st.session_state.analysis_history:
    history_df = pd.DataFrame(st.session_state.analysis_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No analyses yet. Run a sentiment check to populate recent history.")

st.markdown("---")
st.subheader("Batch analysis")
uploaded_file = st.file_uploader("Upload a CSV file with a text column", type=["csv"])
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read the CSV file: {exc}")
        batch_df = None

    if batch_df is not None:
        text_columns = [
            col
            for col in batch_df.columns
            if any(keyword in col.lower() for keyword in ["text", "review", "comment", "sentence", "content"])
        ]
        if not text_columns:
            st.error("The uploaded CSV does not contain a text column. Please include a column like 'text' or 'review'.")
        else:
            text_column = text_columns[0]
            batch_df = batch_df[[text_column]].copy().rename(columns={text_column: "input_text"})
            batch_df = batch_df.head(20)
            batch_results = []

            for _, row in batch_df.iterrows():
                raw_text = str(row["input_text"])
                if not raw_text or not raw_text.strip():
                    batch_results.append({
                        "Input review": "",
                        "NB sentiment": "Unavailable",
                        "LLM sentiment": "Unavailable",
                        "Agreement": "N/A",
                    })
                    continue

                try:
                    nb_result = predict_sentiment(raw_text)
                    llm_result = analyze_text(raw_text, api_key=api_key, model_name=model_name)
                    comparison = compare_predictions(nb_result, llm_result)
                    batch_results.append({
                        "Input review": raw_text,
                        "NB sentiment": nb_result["predicted_sentiment"],
                        "LLM sentiment": llm_result.get("sentiment", "Unavailable"),
                        "Agreement": "Agree" if comparison["agree"] else "Disagree",
                    })
                except Exception as exc:
                    batch_results.append({
                        "Input review": raw_text,
                        "NB sentiment": "Error",
                        "LLM sentiment": "Error",
                        "Agreement": "N/A",
                    })
                    st.warning(f"Failed to analyse one row: {exc}")

            batch_results_df = pd.DataFrame(batch_results)
            st.dataframe(batch_results_df, use_container_width=True, hide_index=True)

            csv_data = batch_results_df.to_csv(index=False)
            st.download_button(
                label="Download batch results as CSV",
                data=csv_data,
                file_name="sentiment_batch_results.csv",
                mime="text/csv",
            )

st.markdown("---")
