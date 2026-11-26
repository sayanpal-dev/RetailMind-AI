from pathlib import Path
import csv

import pandas as pd

from day14_llm_client import (
    generate_text
)

from day14_prompts import (
    build_review_summary_prompt
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


REVIEWS_PATH = (
    PROJECT_ROOT /
    "data" /
    "day12" /
    "amazon_reviews.csv"
)


OUTPUT_DIRECTORY = (
    PROJECT_ROOT /
    "outputs" /
    "day14"
)


SUMMARY_PATH = (
    OUTPUT_DIRECTORY /
    "review_summary.txt"
)


DETAILS_PATH = (
    OUTPUT_DIRECTORY /
    "summary_details.csv"
)


MAXIMUM_REVIEWS = 40

REVIEWS_PER_BATCH = 10


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Possible text columns
# --------------------------------------------------

TEXT_CANDIDATES = [
    "text",
    "review",
    "reviewText",
    "review_body",
    "content",
    "sentence"
]


def find_text_column(
    columns
):

    for candidate in TEXT_CANDIDATES:

        if candidate in columns:

            return candidate

    return None


# --------------------------------------------------
# 3. Load review text
# --------------------------------------------------

def load_reviews():

    if not REVIEWS_PATH.exists():

        raise FileNotFoundError(
            "\nAmazon review data not found:\n"
            "data/day12/amazon_reviews.csv"
        )

    dataframe = pd.read_csv(
        REVIEWS_PATH,
        encoding_errors="ignore",
        on_bad_lines="skip"
    )

    text_column = find_text_column(
        dataframe.columns
    )

    if text_column is None:

        raise ValueError(
            "No review-text column was found."
        )

    reviews = (
        dataframe[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    reviews = reviews[
        reviews.str.len() > 5
    ]

    reviews = reviews.drop_duplicates()

    reviews = reviews.head(
        MAXIMUM_REVIEWS
    )

    if len(reviews) == 0:

        raise ValueError(
            "No valid reviews were loaded."
        )

    return reviews.tolist()


# --------------------------------------------------
# 4. Split reviews into batches
# --------------------------------------------------

def create_batches(
    reviews,
    batch_size
):

    return [
        reviews[
            index:
            index + batch_size
        ]
        for index in range(
            0,
            len(reviews),
            batch_size
        )
    ]


# --------------------------------------------------
# 5. Summarize multiple reviews
# --------------------------------------------------

def summarize_reviews():

    reviews = load_reviews()

    batches = create_batches(
        reviews,
        REVIEWS_PER_BATCH
    )

    batch_summaries = []

    detail_rows = []

    print("=" * 60)

    print(
        "DAY 14: REVIEW THEME SUMMARIZATION"
    )

    print("=" * 60)

    print(
        "Reviews selected:",
        len(reviews)
    )

    print(
        "Summary batches:",
        len(batches)
    )

    for batch_number, batch in enumerate(
        batches,
        start=1
    ):

        print(
            f"\nSummarizing batch "
            f"{batch_number}/"
            f"{len(batches)}..."
        )

        prompt = (
            build_review_summary_prompt(
                batch
            )
        )

        batch_summary = generate_text(
            prompt
        )

        batch_summaries.append(
            batch_summary
        )

        detail_rows.append({
            "batch":
                batch_number,

            "review_count":
                len(batch),

            "summary":
                batch_summary
        })

    # Combine the smaller summaries into
    # one final product-level summary.
    final_prompt = (
        "You are RetailMind's review analyst.\n\n"
        "Combine the following batch summaries into "
        "one concise product-level summary.\n\n"
        "Do not invent information.\n"
        "Remove repeated themes.\n"
        "Clearly separate positive themes, negative "
        "themes, recurring issues and recommended "
        "business actions.\n\n"
        "BATCH SUMMARIES:\n\n" +
        "\n\n".join(
            batch_summaries
        )
    )

    final_summary = generate_text(
        final_prompt
    )

    SUMMARY_PATH.write_text(
        final_summary,
        encoding="utf-8"
    )

    with open(
        DETAILS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "batch",
                "review_count",
                "summary"
            ]
        )

        writer.writeheader()

        writer.writerows(
            detail_rows
        )

    print("\n" + "=" * 60)

    print("FINAL REVIEW SUMMARY")

    print("=" * 60)

    print(final_summary)

    print(
        "\nSummary saved:",
        SUMMARY_PATH
    )

    print(
        "Batch details saved:",
        DETAILS_PATH
    )

    return final_summary


# --------------------------------------------------
# 6. Run directly
# --------------------------------------------------

if __name__ == "__main__":

    summarize_reviews()