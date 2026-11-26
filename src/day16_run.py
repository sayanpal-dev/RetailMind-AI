from pathlib import Path
import subprocess
import sys


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


VARIATION_SCRIPT = (
    PROJECT_ROOT /
    "src" /
    "day16_generate_variations.py"
)


VIDEO_SCRIPT = (
    PROJECT_ROOT /
    "src" /
    "day16_create_video.py"
)


# --------------------------------------------------
# 2. Run one script
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Generate variations
# --------------------------------------------------

run_script(
    VARIATION_SCRIPT
)


# --------------------------------------------------
# 4. Create promotional video
# --------------------------------------------------

run_script(
    VIDEO_SCRIPT
)


# --------------------------------------------------
# 5. Finish
# --------------------------------------------------

print("\n" + "=" * 60)

print("DAY 16 COMPLETED SUCCESSFULLY")

print("=" * 60)