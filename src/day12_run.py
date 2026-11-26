from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


TRAIN_SCRIPT = (
    PROJECT_ROOT /
    "src" /
    "day12_train_sentiment.py"
)


PREDICT_SCRIPT = (
    PROJECT_ROOT /
    "src" /
    "day12_predict_sentiment.py"
)


def run_command(command):

    print("\nRunning:")

    print(
        " ".join(
            str(part)
            for part in command
        )
    )

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True
    )


# --------------------------------------------------
# 1. Train and evaluate
# --------------------------------------------------

run_command([
    sys.executable,
    str(TRAIN_SCRIPT)
])


# --------------------------------------------------
# 2. Positive example
# --------------------------------------------------

run_command([
    sys.executable,
    str(PREDICT_SCRIPT),
    (
        "The product is excellent, "
        "works perfectly and is great value"
    )
])


# --------------------------------------------------
# 3. Negative example
# --------------------------------------------------

run_command([
    sys.executable,
    str(PREDICT_SCRIPT),
    (
        "The item arrived broken and "
        "was a complete waste of money"
    )
])


print("\n" + "=" * 55)

print("DAY 12 COMPLETED SUCCESSFULLY")

print("=" * 55)