# --------------------------------------------------
# Day 14 prompt templates
# --------------------------------------------------


SUMMARY_SYSTEM_PROMPT = """
You are RetailMind's product-review analyst.

Your responsibility is to summarize customer reviews accurately.

Rules:
1. Use only the supplied customer reviews.
2. Do not invent product details.
3. Identify recurring positive themes.
4. Identify recurring negative themes.
5. Identify common customer requests or product issues.
6. Mention mixed or conflicting feedback.
7. Keep the summary concise and useful for a retail manager.
8. Do not reproduce personal information from reviews.
"""


SUPPORT_SYSTEM_PROMPT = """
You are RetailMind's customer-support assistant.

Rules:
1. Answer only from the supplied product information and support policy.
2. Do not invent prices, warranties, return periods or delivery promises.
3. If the context does not contain the answer, clearly say so.
4. Recommend human support when the context is insufficient.
5. Keep the answer concise, professional and helpful.
6. Use conversation history only to understand follow-up questions.
7. Product and policy information always take priority over conversation history.
"""


def build_review_summary_prompt(
    reviews
):
    """
    Build the final prompt used to summarize customer reviews.
    """

    formatted_reviews = []

    for index, review in enumerate(
        reviews,
        start=1
    ):
        formatted_reviews.append(
            f"Review {index}: {review}"
        )

    reviews_text = "\n".join(
        formatted_reviews
    )

    return f"""
{SUMMARY_SYSTEM_PROMPT}

Customer reviews:

{reviews_text}

Return the response in this format:

Overall sentiment:
<one short paragraph>

Common positive themes:
- Theme 1
- Theme 2

Common negative themes:
- Theme 1
- Theme 2

Recurring product or service issues:
- Issue 1
- Issue 2

Recommended business actions:
- Action 1
- Action 2
"""


def build_support_prompt(
    product_information,
    support_policy,
    conversation_history,
    customer_question
):
    """
    Build the grounded support-assistant prompt.
    """

    history_lines = []

    for message in conversation_history:

        role = message["role"]

        content = message["content"]

        history_lines.append(
            f"{role}: {content}"
        )

    history_text = "\n".join(
        history_lines
    )

    if not history_text:

        history_text = (
            "No previous conversation."
        )

    return f"""
{SUPPORT_SYSTEM_PROMPT}

PRODUCT INFORMATION
-------------------
{product_information}

SUPPORT POLICY
--------------
{support_policy}

CONVERSATION HISTORY
--------------------
{history_text}

CURRENT CUSTOMER QUESTION
-------------------------
{customer_question}

Answer using only the supplied product information and support policy.
"""