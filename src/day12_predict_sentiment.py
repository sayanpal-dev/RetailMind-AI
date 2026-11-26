from pathlib import Path
import sys

import joblib


MODEL_PATH = Path(
    "models/day12_sentiment_pipeline.joblib"
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "\nSentiment pipeline not found:\n"
        "models/day12_sentiment_pipeline.joblib\n\n"
        "Run day12_train_sentiment.py first."
    )


pipeline = joblib.load(
    MODEL_PATH
)


def predict_sentiment(review):

    review = str(
        review
    ).strip()

    if not review:

        raise ValueError(
            "Review cannot be empty."
        )

    prediction = int(
        pipeline.predict(
            [review]
        )[0]
    )

    probabilities = (
        pipeline.predict_proba(
            [review]
        )[0]
    )

    sentiment = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    confidence = float(
        probabilities[prediction]
    )

    return {
        "review": review,
        "sentiment": sentiment,
        "confidence": round(
            confidence,
            4
        )
    }


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:\n"
            "python src/day12_predict_sentiment.py "
            "\"your customer review\""
        )

        raise SystemExit(1)

    input_review = " ".join(
        sys.argv[1:]
    )

    result = predict_sentiment(
        input_review
    )

    print("\n" + "=" * 55)
    print("DAY 12 SENTIMENT PREDICTION")
    print("=" * 55)

    print(
        "Review    :",
        result["review"]
    )

    print(
        "Sentiment :",
        result["sentiment"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )