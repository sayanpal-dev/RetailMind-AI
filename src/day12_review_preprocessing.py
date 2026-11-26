import re

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer


tokenizer = TreebankWordTokenizer()

stemmer = PorterStemmer()


STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "if", "while", "of", "at", "by", "for",
    "with", "about", "against", "between",
    "into", "through", "during", "before",
    "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further",
    "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor",
    "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just",
    "is", "am", "are", "was", "were", "be",
    "been", "being", "have", "has", "had",
    "do", "does", "did", "this", "that",
    "these", "those", "i", "me", "my",
    "myself", "we", "our", "ours", "you",
    "your", "yours", "he", "him", "his",
    "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs"
}


def clean_review(text):
    """
    Clean one product review.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove HTML
    4. Remove numbers and punctuation
    5. Tokenize
    6. Remove stopwords
    7. Apply stemming
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Remove URLs.
    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Retain only letters and apostrophes.
    text = re.sub(
        r"[^a-z']+",
        " ",
        text
    )

    tokens = tokenizer.tokenize(
        text
    )

    cleaned_tokens = []

    for token in tokens:

        token = token.strip("'")

        if len(token) < 2:
            continue

        # Preserve important negation words.
        if (
            token in STOPWORDS
            and token not in {
                "not",
                "no",
                "never"
            }
        ):
            continue

        cleaned_tokens.append(
            stemmer.stem(token)
        )

    return " ".join(
        cleaned_tokens
    )