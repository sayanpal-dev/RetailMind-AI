from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PRODUCT_DATA_PATH = Path(
    "data/day15/products.csv"
)

PROMPT_LIBRARY_PATH = Path(
    "models/day15_prompt_library.csv"
)


# --------------------------------------------------
# 2. Default negative prompt
# --------------------------------------------------

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, "
    "bad anatomy, duplicate product, cropped, "
    "out of frame, watermark, signature, logo, "
    "misspelled text, unreadable text, cluttered "
    "background, oversaturated, underexposed"
)


# --------------------------------------------------
# 3. Build one prompt
# --------------------------------------------------

def build_marketing_prompt(
    product_name,
    category,
    style,
    setting,
    brand_colors,
    campaign
):

    prompt = (
        f"Professional commercial product photography "
        f"of a {category} named {product_name}, "
        f"{style} visual style, "
        f"displayed in {setting}, "
        f"brand color palette of {brand_colors}, "
        f"created for a {campaign}, "
        f"realistic product materials, "
        f"premium retail advertising, "
        f"clean composition, dramatic studio lighting, "
        f"sharp focus, highly detailed, photorealistic"
    )

    return prompt


# --------------------------------------------------
# 4. Load products
# --------------------------------------------------

def load_products():

    if not PRODUCT_DATA_PATH.exists():

        raise FileNotFoundError(
            "\nProduct metadata was not found:\n"
            "data/day15/products.csv"
        )

    products = pd.read_csv(
        PRODUCT_DATA_PATH
    )

    required_columns = [
        "product_name",
        "category",
        "style",
        "setting",
        "brand_colors",
        "campaign"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in products.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    products = products.dropna(
        subset=required_columns
    ).copy()

    return products


# --------------------------------------------------
# 5. Create complete prompt library
# --------------------------------------------------

def create_prompt_library():

    products = load_products()

    prompt_rows = []

    for _, row in products.iterrows():

        prompt = build_marketing_prompt(
            product_name=row["product_name"],
            category=row["category"],
            style=row["style"],
            setting=row["setting"],
            brand_colors=row["brand_colors"],
            campaign=row["campaign"]
        )

        prompt_rows.append({
            "product_name":
                row["product_name"],

            "category":
                row["category"],

            "prompt":
                prompt,

            "negative_prompt":
                DEFAULT_NEGATIVE_PROMPT,

            "seed":
                42,

            "guidance_scale":
                7.5,

            "inference_steps":
                25
        })

    prompt_library = pd.DataFrame(
        prompt_rows
    )

    PROMPT_LIBRARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    prompt_library.to_csv(
        PROMPT_LIBRARY_PATH,
        index=False
    )

    print(
        "\nPrompt library created:",
        PROMPT_LIBRARY_PATH
    )

    print(
        prompt_library[
            [
                "product_name",
                "category",
                "seed"
            ]
        ].to_string(
            index=False
        )
    )

    return prompt_library


# --------------------------------------------------
# 6. Run directly
# --------------------------------------------------

if __name__ == "__main__":

    create_prompt_library()