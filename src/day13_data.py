from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/day12/amazon_reviews.csv"
)


TEXT_CANDIDATES = [
    "text",
    "review",
    "reviewText",
    "review_body",
    "content",
    "sentence"
]


LABEL_CANDIDATES = [
    "label",
    "sentiment",
    "polarity",
    "target",
    "class"
]


POSITIVE_LABELS = {
    "1",
    "2",
    "positive",
    "pos",
    "good"
}


NEGATIVE_LABELS = {
    "0",
    "negative",
    "neg",
    "bad"
}


def find_column(
    columns,
    candidates
):

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


def normalize_sentiment(label):

    label = str(
        label
    ).strip().lower()

    if label in POSITIVE_LABELS:
        return 1

    if label in NEGATIVE_LABELS:
        return 0

    return None


def load_review_data(
    maximum_rows=3000
):

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "\nAmazon review data not found:\n"
            "data/day12/amazon_reviews.csv"
        )

    df = pd.read_csv(
        DATA_PATH,
        encoding_errors="ignore",
        on_bad_lines="skip"
    )

    text_column = find_column(
        df.columns,
        TEXT_CANDIDATES
    )

    label_column = find_column(
        df.columns,
        LABEL_CANDIDATES
    )

    if text_column is None:

        raise ValueError(
            "No review-text column was found."
        )

    if label_column is None:

        raise ValueError(
            "No sentiment-label column was found."
        )

    prepared = pd.DataFrame({
        "text": (
            df[text_column]
            .fillna("")
            .astype(str)
            .str.strip()
        ),

        "label": (
            df[label_column]
            .apply(
                normalize_sentiment
            )
        )
    })

    prepared = prepared.dropna(
        subset=[
            "text",
            "label"
        ]
    )

    prepared = prepared[
        prepared["text"].str.len() > 2
    ]

    prepared["label"] = (
        prepared["label"]
        .astype(int)
    )

    # Balance positive and negative reviews
    grouped_data = []

    category_limit = (
        maximum_rows // 2
    )

    for label_value in [0, 1]:

        label_data = prepared[
            prepared["label"] ==
            label_value
        ]

        number_to_take = min(
            category_limit,
            len(label_data)
        )

        grouped_data.append(
            label_data.sample(
                n=number_to_take,
                random_state=42
            )
        )

    prepared = pd.concat(
        grouped_data,
        ignore_index=True
    )

    prepared = prepared.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )

    if len(prepared) < 40:

        raise ValueError(
            "At least 40 valid reviews are "
            "required."
        )

    return prepared