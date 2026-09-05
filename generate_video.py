"""Generate a short video from an input image with Wan 2.2."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Sequence


MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_IMAGE_PATH = Path("inputs/input.jpg")
DEFAULT_OUTPUT_PATH = Path("outputs/output.mp4")
DEFAULT_PROMPT = (
    "The person slowly turns their head toward the camera and smiles slightly. "
    "The camera remains stationary. Natural movement, realistic lighting."
)
NEGATIVE_PROMPT = (
    "oversaturated, overexposed, static, blurry details, subtitles, stylized, "
    "artwork, painting, low quality, JPEG artifacts, deformed, disfigured, "
    "extra fingers, poorly drawn hands, poorly drawn face, malformed limbs, "
    "fused fingers, cluttered background, duplicate subject, walking backwards"
)

TARGET_AREA = 1280 * 704
NUM_FRAMES = 121
FPS = 24
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
DEFAULT_SEED = 42


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a five-second video from an image with Wan 2.2."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"source image (default: {DEFAULT_IMAGE_PATH})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="description of the desired motion",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output MP4 path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args(argv)


def calculate_dimensions(
    image_width: int,
    image_height: int,
    multiple: int,
    max_area: int = TARGET_AREA,
) -> tuple[int, int]:
    """Fit the source aspect ratio to the target area and model dimensions."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if multiple <= 0:
        raise ValueError("Dimension multiple must be positive.")

    aspect_ratio = image_height / image_width
    height = round(math.sqrt(max_area * aspect_ratio)) // multiple * multiple
    width = round(math.sqrt(max_area / aspect_ratio)) // multiple * multiple
    return max(height, multiple), max(width, multiple)


def gibibytes(byte_count: int) -> float:
    """Convert a byte count to GiB."""
    return byte_count / (1024**3)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the image-to-video generation workflow."""
    args = parse_args(argv)
    total_started = time.perf_counter()

    if not args.image.is_file():
        raise SystemExit(f"Input image does not exist: {args.image}")
    if not args.prompt.strip():
        raise SystemExit("Prompt must not be empty.")

    # Keep heavyweight imports after input validation so --help and common
    # mistakes do not initialize PyTorch or load model-related libraries.
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-capable GPU is required for video generation.")

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_properties(device)
    torch.cuda.reset_peak_memory_stats(device)

    image = load_image(str(args.image)).convert("RGB")

    print("Wan 2.2 image-to-video generation")
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {gpu.name} ({gibibytes(gpu.total_memory):.1f} GiB VRAM)")
    print(f"Input: {args.image} ({image.width}x{image.height})")
    print(f"Output: {args.output}")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    print(f"Frames: {NUM_FRAMES} at {FPS} FPS ({(NUM_FRAMES - 1) / FPS:.1f}s)")
    print(f"Inference steps: {NUM_INFERENCE_STEPS}")
    print(f"Guidance scale: {GUIDANCE_SCALE}")
    print("Memory mode: model CPU offload")

    print("Loading model...", flush=True)
    model_load_started = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        dtype=torch.float32,
    )
    pipeline = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload(device="cuda")
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_started

    spatial_multiple = (
        pipeline.vae_scale_factor_spatial
        * pipeline.transformer.config.patch_size[1]
    )
    height, width = calculate_dimensions(
        image.width,
        image.height,
        spatial_multiple,
    )
    print(f"Resolution: {width}x{height}")
    print(f"Model loading time: {model_load_seconds:.1f}s")
    print(
        "VRAM after loading: "
        f"{gibibytes(torch.cuda.memory_allocated(device)):.1f} GiB allocated, "
        f"{gibibytes(torch.cuda.memory_reserved(device)):.1f} GiB reserved"
    )

    torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    print("Generating video...", flush=True)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        frames = pipeline(
            image=image,
            prompt=args.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).frames[0]
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    peak_vram = gibibytes(torch.cuda.max_memory_allocated(device))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Exporting MP4...", flush=True)
    export_started = time.perf_counter()
    export_to_video(frames, str(args.output), fps=FPS)
    export_seconds = time.perf_counter() - export_started
    total_seconds = time.perf_counter() - total_started

    print(f"Inference time: {inference_seconds:.1f}s")
    print(f"Peak inference VRAM: {peak_vram:.1f} GiB")
    print(f"MP4 export time: {export_seconds:.1f}s")
    print(f"Total execution time: {total_seconds:.1f}s")
    print(f"Saved video: {args.output}")


if __name__ == "__main__":
    main()
