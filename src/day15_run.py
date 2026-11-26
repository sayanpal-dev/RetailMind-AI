from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


PROMPT_SCRIPT = (
    PROJECT_ROOT /
    "src" /
    "day15_prompt_library.py"
)


CAMPAIGN_SCRIPT = (
    PROJECT_ROOT /
    "src" /
    "day15_generate_campaign.py"
)


def run_script(
    script_path
):

    print("\nRunning:")

    print(
        script_path
    )

    subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        cwd=PROJECT_ROOT,
        check=True
    )


# Create prompt library.
run_script(
    PROMPT_SCRIPT
)


# Generate marketing images.
run_script(
    CAMPAIGN_SCRIPT
)


print("\n" + "=" * 60)

print("DAY 15 COMPLETED SUCCESSFULLY")

print("=" * 60)