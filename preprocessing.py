import re

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def clean_text(text: str) -> str:
    if text is None:
        return ""

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    cleaned = cleaned.replace("&amp;", "&")
    cleaned = re.sub(r"http\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"@\w+|#\w+", " ", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def label_for_index(index: int) -> str:
    return LABELS.get(int(index), "Neutral")
