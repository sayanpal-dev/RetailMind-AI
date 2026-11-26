from pathlib import Path
import re

import torch

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_ID = (
    "stable-diffusion-v1-5/"
    "stable-diffusion-v1-5"
)

OUTPUT_DIRECTORY = Path(
    "outputs/day15_marketing"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Safe filename
# --------------------------------------------------

def safe_filename(text):

    text = str(
        text
    ).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


# --------------------------------------------------
# 3. Select device
# --------------------------------------------------

def get_device():

    if torch.cuda.is_available():

        return (
            "cuda",
            torch.float16
        )

    return (
        "cpu",
        torch.float32
    )


# --------------------------------------------------
# 4. Load pipeline
# --------------------------------------------------

def load_diffusion_pipeline():

    device, data_type = get_device()

    print("\nLoading Stable Diffusion...")
    print("Device:", device)

    pipeline = (
        StableDiffusionPipeline
        .from_pretrained(
            MODEL_ID,
            torch_dtype=data_type,
            use_safetensors=True,
            safety_checker=None
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

    if device == "cuda":

        pipeline.enable_attention_slicing()

    else:

        print(
            "\nWARNING: CUDA GPU was not found."
        )

        print(
            "CPU generation may take a long time."
        )

    return pipeline, device


# --------------------------------------------------
# 5. Generate one image
# --------------------------------------------------

def generate_marketing_image(
    pipeline,
    product_name,
    prompt,
    negative_prompt,
    seed=42,
    guidance_scale=7.5,
    inference_steps=25,
    width=512,
    height=512
):

    device, _ = get_device()

    generator = torch.Generator(
        device=device
    ).manual_seed(
        int(seed)
    )

    result = pipeline(
        prompt=str(prompt),
        negative_prompt=str(
            negative_prompt
        ),
        num_inference_steps=int(
            inference_steps
        ),
        guidance_scale=float(
            guidance_scale
        ),
        width=int(width),
        height=int(height),
        generator=generator
    )

    image = result.images[0]

    output_name = (
        f"{safe_filename(product_name)}_"
        f"{int(seed)}.png"
    )

    output_path = (
        OUTPUT_DIRECTORY /
        output_name
    )

    image.save(
        output_path
    )

    return output_path