from pathlib import Path
import os
import time

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ENV_PATH = (
    PROJECT_ROOT /
    ".env"
)


# --------------------------------------------------
# 2. Load environment variables
# --------------------------------------------------

load_dotenv(
    ENV_PATH
)


API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

MODEL_NAME = os.getenv(
    "GEMINI_MODEL"
)


# --------------------------------------------------
# 3. Validate configuration
# --------------------------------------------------

if not API_KEY:

    raise ValueError(
        "\nGEMINI_API_KEY was not found.\n\n"
        "Create a .env file in the project root:\n"
        "GEMINI_API_KEY=your_api_key\n"
        "GEMINI_MODEL=your_available_model"
    )


if not MODEL_NAME:

    raise ValueError(
        "\nGEMINI_MODEL was not found.\n\n"
        "Add this to the project .env file:\n"
        "GEMINI_MODEL=your_available_model"
    )


# --------------------------------------------------
# 4. Create Gemini client
# --------------------------------------------------

client = genai.Client(
    api_key=API_KEY
)


# --------------------------------------------------
# 5. Text-generation helper
# --------------------------------------------------

def generate_text(
    prompt,
    retries=3
):
    """
    Send a prompt to Gemini and return text.

    Retries temporary failures such as a
    503 high-demand response.
    """

    prompt = str(
        prompt
    ).strip()

    if not prompt:

        raise ValueError(
            "Prompt cannot be empty."
        )

    last_error = None

    for attempt in range(
        1,
        retries + 1
    ):

        try:

            response = (
                client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
            )

            response_text = getattr(
                response,
                "text",
                None
            )

            if not response_text:

                raise RuntimeError(
                    "The LLM returned no text."
                )

            return response_text.strip()

        except Exception as error:

            last_error = error

            if attempt < retries:

                wait_seconds = (
                    attempt * 5
                )

                print(
                    f"\nLLM request failed. "
                    f"Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(
                    wait_seconds
                )

    raise RuntimeError(
        "The LLM request failed after "
        f"{retries} attempts.\n"
        f"Last error: {last_error}"
    )