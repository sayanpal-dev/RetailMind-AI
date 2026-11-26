from pathlib import Path
import csv

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer
)
from sklearn.linear_model import (
    LogisticRegression
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from day12_review_preprocessing import (
    clean_review
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_FEATURES = 15000

DATA_PATH = Path(
    "data/day12/amazon_reviews.csv"
)

MODEL_DIRECTORY = Path(
    "models"
)

MODEL_PATH = (
    MODEL_DIRECTORY /
    "day12_sentiment_pipeline.joblib"
)

RESULTS_PATH = (
    MODEL_DIRECTORY /
    "day12_results.csv"
)

TERMS_PATH = (
    MODEL_DIRECTORY /
    "day12_informative_terms.csv"
)

MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Recognized column names
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Find a matching column
# --------------------------------------------------

def find_column(
    columns,
    candidates
):

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


# --------------------------------------------------
# 4. Normalize labels
# --------------------------------------------------

def normalize_label(label):

    label = str(
        label
    ).strip().lower()

    if label in POSITIVE_LABELS:
        return 1

    if label in NEGATIVE_LABELS:
        return 0

    return None


# --------------------------------------------------
# 5. Check dataset
# --------------------------------------------------

if not DATA_PATH.exists():

    raise FileNotFoundError(
        "\nDataset not found:\n"
        "data/day12/amazon_reviews.csv\n\n"
        "Put the Amazon reviews CSV inside "
        "the data/day12 folder."
    )


# --------------------------------------------------
# 6. Read dataset
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    encoding_errors="ignore",
    on_bad_lines="skip"
)

print("=" * 60)
print("DAY 12: CLASSICAL SENTIMENT ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 7. Find text and label columns
# --------------------------------------------------

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
        "No text column was found.\n"
        "Rename the review column to 'text' "
        "or add its name to TEXT_CANDIDATES."
    )


if label_column is None:

    raise ValueError(
        "No label column was found.\n"
        "Rename the sentiment column to 'label' "
        "or add its name to LABEL_CANDIDATES."
    )


print("\nDetected text column :", text_column)
print("Detected label column:", label_column)


# --------------------------------------------------
# 8. Prepare reviews
# --------------------------------------------------

prepared_df = pd.DataFrame({
    "text": (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    ),

    "label": (
        df[label_column]
        .apply(normalize_label)
    )
})


prepared_df = prepared_df.dropna(
    subset=[
        "text",
        "label"
    ]
)

prepared_df = prepared_df[
    prepared_df["text"].str.len() > 2
].copy()

prepared_df["label"] = (
    prepared_df["label"]
    .astype(int)
)


if len(prepared_df) < 20:

    raise ValueError(
        "Fewer than 20 valid reviews were loaded."
    )


if prepared_df["label"].nunique() < 2:

    raise ValueError(
        "Both positive and negative reviews "
        "are required."
    )


print("\nValid reviews:")
print(len(prepared_df))

print("\nLabel counts:")
print(
    prepared_df["label"]
    .map({
        0: "Negative",
        1: "Positive"
    })
    .value_counts()
)


# --------------------------------------------------
# 9. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        prepared_df["text"],
        prepared_df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=prepared_df["label"]
    )
)


print("\nTraining reviews:", len(X_train))
print("Testing reviews :", len(X_test))


# --------------------------------------------------
# 10. Create pipeline
# --------------------------------------------------

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            preprocessor=clean_review,
            lowercase=False,
            max_features=MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
])


# --------------------------------------------------
# 11. Train
# --------------------------------------------------

print("\nTraining TF-IDF sentiment model...")

pipeline.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 12. Predict
# --------------------------------------------------

predictions = pipeline.predict(
    X_test
)


# --------------------------------------------------
# 13. Evaluate
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[
        0,
        1
    ]
)


print("\n" + "=" * 60)
print("DAY 12 RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion matrix:")

print(matrix)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Negative",
            "Positive"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# 14. Save complete pipeline
# --------------------------------------------------

joblib.dump(
    pipeline,
    MODEL_PATH
)


# --------------------------------------------------
# 15. Save result CSV
# --------------------------------------------------

results_df = pd.DataFrame([
    {
        "Model":
            "TF-IDF + Logistic Regression",

        "Training Reviews":
            len(X_train),

        "Testing Reviews":
            len(X_test),

        "Accuracy":
            round(
                accuracy,
                4
            ),

        "Precision":round(float(precision), 4),
        "Recall":round(float(recall),4),

        "F1":round(float(f1), 4),

        "True Negative":
            int(matrix[0][0]),

        "False Positive":
            int(matrix[0][1]),

        "False Negative":
            int(matrix[1][0]),

        "True Positive":
            int(matrix[1][1])
    }
])


results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# --------------------------------------------------
# 16. Extract informative terms
# --------------------------------------------------

vectorizer = pipeline.named_steps[
    "tfidf"
]

classifier = pipeline.named_steps[
    "classifier"
]

feature_names = (
    vectorizer.get_feature_names_out()
)

coefficients = (
    classifier.coef_[0]
)


term_pairs = list(
    zip(
        feature_names,
        coefficients
    )
)


negative_terms = sorted(
    term_pairs,
    key=lambda pair: pair[1]
)[:20]


positive_terms = sorted(
    term_pairs,
    key=lambda pair: pair[1],
    reverse=True
)[:20]


term_rows = []


for term, weight in negative_terms:

    term_rows.append({
        "Sentiment": "Negative",
        "Term": term,
        "Model Weight": round(
            float(weight),
            4
        )
    })


for term, weight in positive_terms:

    term_rows.append({
        "Sentiment": "Positive",
        "Term": term,
        "Model Weight": round(
            float(weight),
            4
        )
    })


pd.DataFrame(
    term_rows
).to_csv(
    TERMS_PATH,
    index=False
)


# --------------------------------------------------
# 17. Finish
# --------------------------------------------------

print("\nGenerated files:")

print("Pipeline:", MODEL_PATH)
print("Results :", RESULTS_PATH)
print("Terms   :", TERMS_PATH)