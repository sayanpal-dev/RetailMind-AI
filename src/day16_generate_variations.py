import time

import pandas as pd
import torch

from PIL import Image, ImageOps

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler
)

from day16_config import (
    INPUT_IMAGE_PATH,
    VARIATION_DIRECTORY,
    RESULTS_PATH,
    MODEL_ID,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    INFERENCE_STEPS,
    SEED,
    PROMPT,
    NEGATIVE_PROMPT,
    VARIATION_SETTINGS
)


# --------------------------------------------------
# 1. Create output folders
# --------------------------------------------------

VARIATION_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Check input image
# --------------------------------------------------

if not INPUT_IMAGE_PATH.exists():

    raise FileNotFoundError(
        "\nDay 16 input image not found:\n"
        "data/day16/input_product.jpg\n\n"
        "Copy one Myntra product image into "
        "the data/day16 folder and rename it "
        "input_product.jpg."
    )


# --------------------------------------------------
# 3. Select device
# --------------------------------------------------

if torch.cuda.is_available():

    device = "cuda"

    data_type = torch.float16

else:

    device = "cpu"

    data_type = torch.float32


print("=" * 60)

print("DAY 16: PRODUCT IMAGE VARIATIONS")

print("=" * 60)

print("Device:", device)


if device == "cpu":

    print(
        "\nWARNING: CUDA GPU was not found."
    )

    print(
        "Stable Diffusion generation on CPU "
        "can be extremely slow."
    )


# --------------------------------------------------
# 4. Prepare the input image
# --------------------------------------------------

input_image = Image.open(
    INPUT_IMAGE_PATH
).convert(
    "RGB"
)


# Preserve the product while producing
# a square 512 x 512 model input.
input_image = ImageOps.contain(
    input_image,
    (
        IMAGE_WIDTH,
        IMAGE_HEIGHT
    ),
    method=Image.Resampling.LANCZOS
)


background = Image.new(
    "RGB",
    (
        IMAGE_WIDTH,
        IMAGE_HEIGHT
    ),
    color=(
        245,
        245,
        245
    )
)


image_x = (
    IMAGE_WIDTH -
    input_image.width
) // 2


image_y = (
    IMAGE_HEIGHT -
    input_image.height
) // 2


background.paste(
    input_image,
    (
        image_x,
        image_y
    )
)


input_image = background


# Save the standardized original too.
standardized_input_path = (
    VARIATION_DIRECTORY /
    "original_product.png"
)


input_image.save(
    standardized_input_path
)


# --------------------------------------------------
# 5. Load img2img pipeline
# --------------------------------------------------

print("\nLoading Stable Diffusion img2img...")


pipeline = (
    StableDiffusionImg2ImgPipeline
    .from_pretrained(
        MODEL_ID,
        torch_dtype=data_type,
        use_safetensors=True
    )
)


pipeline.scheduler = (
    DPMSolverMultistepScheduler
    .from_config(
        pipeline.scheduler.config
    )
)


pipeline = pipeline.to(
    device
)


pipeline.enable_attention_slicing()


# --------------------------------------------------
# 6. Generate variations
# --------------------------------------------------

result_rows = []


for variation_number, settings in enumerate(
    VARIATION_SETTINGS,
    start=1
):

    strength = float(
        settings["strength"]
    )

    guidance_scale = float(
        settings["guidance_scale"]
    )


    print("\n" + "-" * 60)

    print(
        f"Variation {variation_number}"
    )

    print(
        "Strength:",
        strength
    )

    print(
        "Guidance scale:",
        guidance_scale
    )


    generator = torch.Generator(
        device=device
    ).manual_seed(
        SEED
    )


    started_at = time.time()


    result = pipeline(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=input_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=(
            INFERENCE_STEPS
        ),
        generator=generator
    )


    generated_image = (
        result.images[0]
    )


    strength_code = int(
        strength * 100
    )

    guidance_code = int(
        guidance_scale * 10
    )


    output_name = (
        f"variation_"
        f"s{strength_code:03d}_"
        f"g{guidance_code:03d}.png"
    )


    output_path = (
        VARIATION_DIRECTORY /
        output_name
    )


    generated_image.save(
        output_path
    )


    generation_seconds = (
        time.time() -
        started_at
    )


    result_rows.append({
        "variation":
            variation_number,

        "strength":
            strength,

        "guidance_scale":
            guidance_scale,

        "seed":
            SEED,

        "inference_steps":
            INFERENCE_STEPS,

        "device":
            device,

        "generation_seconds":
            round(
                float(
                    generation_seconds
                ),
                2
            ),

        "input_image":
            str(INPUT_IMAGE_PATH),

        "output_image":
            str(output_path),

        "prompt":
            PROMPT,

        "negative_prompt":
            NEGATIVE_PROMPT
    })


    print(
        "Saved:",
        output_path
    )


# --------------------------------------------------
# 7. Save experiment details
# --------------------------------------------------

results_dataframe = pd.DataFrame(
    result_rows
)


results_dataframe.to_csv(
    RESULTS_PATH,
    index=False
)


print("\n" + "=" * 60)

print("DAY 16 VARIATIONS COMPLETED")

print("=" * 60)


print(
    results_dataframe[
        [
            "variation",
            "strength",
            "guidance_scale",
            "generation_seconds",
            "output_image"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nResult log:",
    RESULTS_PATH
)