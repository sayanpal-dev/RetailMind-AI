from pathlib import Path
import time

import pandas as pd

from day15_generate_marketing_image import (
    load_diffusion_pipeline,
    generate_marketing_image
)


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROMPT_LIBRARY_PATH = Path(
    "models/day15_prompt_library.csv"
)

RESULTS_PATH = Path(
    "models/day15_generation_results.csv"
)


# --------------------------------------------------
# 2. Check prompt library
# --------------------------------------------------

if not PROMPT_LIBRARY_PATH.exists():

    raise FileNotFoundError(
        "\nPrompt library not found:\n"
        "models/day15_prompt_library.csv\n\n"
        "Run day15_prompt_library.py first."
    )


# --------------------------------------------------
# 3. Load prompts
# --------------------------------------------------

prompt_library = pd.read_csv(
    PROMPT_LIBRARY_PATH
)


# --------------------------------------------------
# 4. Load Stable Diffusion once
# --------------------------------------------------

pipeline, device = (
    load_diffusion_pipeline()
)


# --------------------------------------------------
# 5. Generate campaign images
# --------------------------------------------------

result_rows = []


for _, row in prompt_library.iterrows():

    print("\n" + "=" * 60)

    print(
        "Generating:",
        row["product_name"]
    )

    print("=" * 60)

    started = time.time()

    output_path = generate_marketing_image(
        pipeline=pipeline,
        product_name=row["product_name"],
        prompt=row["prompt"],
        negative_prompt=(
            row["negative_prompt"]
        ),
        seed=int(
            row["seed"]
        ),
        guidance_scale=float(
            row["guidance_scale"]
        ),
        inference_steps=int(
            row["inference_steps"]
        )
    )

    generation_seconds = (
        time.time() -
        started
    )

    result_rows.append({
        "product_name":
            row["product_name"],

        "category":
            row["category"],

        "seed":
            int(row["seed"]),

        "guidance_scale":
            float(
                row["guidance_scale"]
            ),

        "inference_steps":
            int(
                row["inference_steps"]
            ),

        "device":
            device,

        "generation_seconds":
            round(
                float(
                    generation_seconds
                ),
                2
            ),

        "output_file":
            str(output_path),

        "prompt":
            row["prompt"],

        "negative_prompt":
            row["negative_prompt"]
    })

    print(
        "Saved:",
        output_path
    )


# --------------------------------------------------
# 6. Save generation record
# --------------------------------------------------

results = pd.DataFrame(
    result_rows
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    RESULTS_PATH,
    index=False
)


print("\n" + "=" * 60)

print("DAY 15 CAMPAIGN GENERATION COMPLETE")

print("=" * 60)

print(
    results[
        [
            "product_name",
            "device",
            "generation_seconds",
            "output_file"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nResult log:",
    RESULTS_PATH
)