from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from PIL import Image

from day16_config import (
    VARIATION_DIRECTORY,
    VIDEO_PATH,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    SECONDS_PER_IMAGE,
    TRANSITION_SECONDS
)


# --------------------------------------------------
# 1. Create output folder
# --------------------------------------------------

VIDEO_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Locate images
# --------------------------------------------------

original_path = (
    VARIATION_DIRECTORY /
    "original_product.png"
)


variation_paths = sorted(
    VARIATION_DIRECTORY.glob(
        "variation_*.png"
    )
)


image_paths = [
    original_path,
    *variation_paths
]


missing_paths = [
    path
    for path in image_paths
    if not path.exists()
]


if missing_paths:

    raise FileNotFoundError(
        "\nRequired Day 16 images are missing.\n"
        "Run day16_generate_variations.py first."
    )


if len(variation_paths) == 0:

    raise ValueError(
        "No generated variations were found."
    )


# --------------------------------------------------
# 3. Prepare one frame
# --------------------------------------------------

def prepare_image(
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    image = image.resize(
        (
            VIDEO_WIDTH,
            VIDEO_HEIGHT
        ),
        Image.Resampling.LANCZOS
    )


    return np.asarray(
        image,
        dtype=np.uint8
    )


# --------------------------------------------------
# 4. Create subtle zoom frames
# --------------------------------------------------

def create_zoom_frames(
    image_array,
    frame_count,
    maximum_zoom=1.05
):

    frames = []


    original_image = Image.fromarray(
        image_array
    )


    for frame_index in range(
        frame_count
    ):

        if frame_count == 1:

            progress = 0.0

        else:

            progress = (
                frame_index /
                (frame_count - 1)
            )


        zoom = (
            1.0 +
            (
                maximum_zoom - 1.0
            ) *
            progress
        )


        resized_width = int(
            VIDEO_WIDTH * zoom
        )

        resized_height = int(
            VIDEO_HEIGHT * zoom
        )


        resized = original_image.resize(
            (
                resized_width,
                resized_height
            ),
            Image.Resampling.LANCZOS
        )


        left = (
            resized_width -
            VIDEO_WIDTH
        ) // 2


        top = (
            resized_height -
            VIDEO_HEIGHT
        ) // 2


        cropped = resized.crop(
            (
                left,
                top,
                left + VIDEO_WIDTH,
                top + VIDEO_HEIGHT
            )
        )


        frames.append(
            np.asarray(
                cropped,
                dtype=np.uint8
            )
        )


    return frames


# --------------------------------------------------
# 5. Create crossfade frames
# --------------------------------------------------

def create_crossfade(
    first_frame,
    second_frame,
    frame_count
):

    frames = []


    first_float = first_frame.astype(
        np.float32
    )

    second_float = second_frame.astype(
        np.float32
    )


    for frame_index in range(
        frame_count
    ):

        alpha = (
            frame_index + 1
        ) / (
            frame_count + 1
        )


        blended = (
            first_float *
            (
                1.0 - alpha
            ) +
            second_float *
            alpha
        )


        blended = np.clip(
            blended,
            0,
            255
        ).astype(
            np.uint8
        )


        frames.append(
            blended
        )


    return frames


# --------------------------------------------------
# 6. Load all images
# --------------------------------------------------

images = [
    prepare_image(path)
    for path in image_paths
]


hold_frame_count = int(
    VIDEO_FPS *
    SECONDS_PER_IMAGE
)


transition_frame_count = int(
    VIDEO_FPS *
    TRANSITION_SECONDS
)


# --------------------------------------------------
# 7. Assemble promotional frames
# --------------------------------------------------

all_frames = []


for image_index, image in enumerate(
    images
):

    zoom_frames = create_zoom_frames(
        image,
        hold_frame_count
    )


    all_frames.extend(
        zoom_frames
    )


    if image_index < len(images) - 1:

        next_image = images[
            image_index + 1
        ]


        crossfade_frames = (
            create_crossfade(
                zoom_frames[-1],
                next_image,
                transition_frame_count
            )
        )


        all_frames.extend(
            crossfade_frames
        )


# Add a short final hold.
all_frames.extend(
    [all_frames[-1]] *
    VIDEO_FPS
)


# --------------------------------------------------
# 8. Write MP4
# --------------------------------------------------

print("=" * 60)

print("DAY 16: CREATING PROMO VIDEO")

print("=" * 60)

print("Images:", len(images))

print("Frames:", len(all_frames))

print("FPS:", VIDEO_FPS)


writer = imageio.get_writer(
    VIDEO_PATH,
    fps=VIDEO_FPS,
    codec="libx264",
    quality=7,
    pixelformat="yuv420p"
)


try:

    for frame in all_frames:

        writer.append_data(
            frame
        )

finally:

    writer.close()


duration_seconds = (
    len(all_frames) /
    VIDEO_FPS
)


print(
    "\nPromo video saved:",
    VIDEO_PATH
)


print(
    "Duration:",
    round(
        float(
            duration_seconds
        ),
        2
    ),
    "seconds"
)
