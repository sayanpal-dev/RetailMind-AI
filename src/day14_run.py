from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


SUMMARY_SCRIPT = (
    PROJECT_ROOT /
    "src" /
    "day14_summarize_reviews.py"
)


CHATBOT_SCRIPT = (
    PROJECT_ROOT /
    "src" /
    "day14_support_chatbot.py"
)


def run_script(
    script_path
):

    print("\nRunning:")

    print(script_path)

    subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        cwd=PROJECT_ROOT,
        check=True
    )


# --------------------------------------------------
# 1. Summarize review themes
# --------------------------------------------------

run_script(
    SUMMARY_SCRIPT
)


# --------------------------------------------------
# 2. Start support chatbot
# --------------------------------------------------

run_script(
    CHATBOT_SCRIPT
)