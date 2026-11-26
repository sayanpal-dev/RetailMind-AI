from pathlib import Path


# --------------------------------------------------
# 1. Project root
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# --------------------------------------------------
# 2. Paths
# --------------------------------------------------

INPUT_IMAGE_PATH = (
    PROJECT_ROOT /
    "data" /
    "day16" /
    "input_product.jpg"
)


OUTPUT_DIRECTORY = (
    PROJECT_ROOT /
    "outputs" /
    "day16"
)


VARIATION_DIRECTORY = (
    OUTPUT_DIRECTORY /
    "variations"
)


RESULTS_PATH = (
    PROJECT_ROOT /
    "models" /
    "day16_variation_results.csv"
)


VIDEO_PATH = (
    OUTPUT_DIRECTORY /
    "retailmind_promo.mp4"
)


# --------------------------------------------------
# 3. Diffusion model
# --------------------------------------------------

MODEL_ID = (
    "stable-diffusion-v1-5/"
    "stable-diffusion-v1-5"
)


# --------------------------------------------------
# 4. Image parameters
# --------------------------------------------------

IMAGE_WIDTH = 512

IMAGE_HEIGHT = 512

INFERENCE_STEPS = 25

SEED = 42


# --------------------------------------------------
# 5. Brand prompt
# --------------------------------------------------

PROMPT = (
    "professional premium retail product photography, "
    "the same product design and category as the input image, "
    "modern minimal commercial advertising style, "
    "navy blue and white brand palette, "
    "clean studio background, soft realistic shadows, "
    "elegant product placement, sharp focus, "
    "realistic materials, high detail, campaign-ready"
)


NEGATIVE_PROMPT = (
    "low quality, blurry, deformed product, "
    "duplicate product, extra objects, incorrect geometry, "
    "cropped product, out of frame, watermark, logo, "
    "signature, text, letters, distorted material, "
    "cluttered background, oversaturated"
)


# --------------------------------------------------
# 6. Variation sweep
# --------------------------------------------------

VARIATION_SETTINGS = [
    {
        "strength": 0.20,
        "guidance_scale": 6.5
    },
    {
        "strength": 0.40,
        "guidance_scale": 7.5
    },
    {
        "strength": 0.60,
        "guidance_scale": 8.5
    }
]


# --------------------------------------------------
# 7. Video settings
# --------------------------------------------------

VIDEO_WIDTH = 512

VIDEO_HEIGHT = 512

VIDEO_FPS = 12

SECONDS_PER_IMAGE = 2

TRANSITION_SECONDS = 1