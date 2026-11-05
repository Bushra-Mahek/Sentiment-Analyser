from typing import Any, Dict


def compare_predictions(nb_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
    nb_label = nb_result.get("predicted_sentiment", "Neutral")
    llm_label = llm_result.get("sentiment", "Unavailable")
    models_agree = nb_label == llm_label

    if llm_result.get("error") or llm_label == "Unavailable":
        summary = "Gemini analysis unavailable"
        explanation = (
            "Naïve Bayes produced a result using the trained model, but Gemini analysis is unavailable "
            "because the API key is missing or the service request failed. The models cannot be compared "
            "until Gemini returns a valid result."
        )
        return {
            "agree": models_agree,
            "summary": summary,
            "nb_prediction": nb_label,
            "llm_prediction": llm_label,
            "difference": "Gemini unavailable",
            "explanation": explanation,
        }

    score_key = max(llm_result.get("scores", {}).items(), key=lambda item: item[1])[0]
    top_score = max(llm_result.get("scores", {}).values(), default=0.0)

    if models_agree:
        summary = "Models Agree"
        explanation = (
            f"Both methods interpreted the review as {nb_label}. The Naïve Bayes classifier assigned a "
            f"confidence of {nb_result.get('confidence', 0.0):.2f}% and Gemini also emphasized {score_key.title()} "
            f"as the strongest estimated score ({top_score:.2f}%)."
        )
    else:
        summary = "Models Disagree"
        explanation = (
            f"Naïve Bayes leaned toward {nb_label}, while Gemini interpreted the text as {llm_label}. This can happen when "
            "the lexical pattern is strongly supported by the training data but the LLM is weighting pragmatic cues such as tone, "
            "sarcasm, negation, or context differently from the model-based classifier."
        )

    return {
        "agree": models_agree,
        "summary": summary,
        "nb_prediction": nb_label,
        "llm_prediction": llm_label,
        "difference": f"Naïve Bayes: {nb_label} | Gemini: {llm_label}",
        "explanation": explanation,
    }
