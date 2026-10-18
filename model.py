
from functools import lru_cache
from transformers import pipeline

CATEGORIES = [
    "symptoms",
    "diagnosis",
    "treatment",
    "medication"
]

MODEL_NAME = "facebook/bart-large-mnli"


@lru_cache(maxsize=1)
def get_classifier():
    return pipeline(
        task="zero-shot-classification",
        model=MODEL_NAME
    )


def classify_text(text: str) -> dict:
    classifier = get_classifier()

    result = classifier(
        text,
        candidate_labels=CATEGORIES,
        multi_label=False
    )

    return {
        "text": text,
        "label": result["labels"][0],
        "score": round(result["scores"][0], 4)
    }


if __name__ == "__main__":
    example = "The patient reports persistent headaches."
    print(classify_text(example))
