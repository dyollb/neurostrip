import argparse
from pathlib import Path

from .lib import is_cuda_available, predict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict brain mask from a 3D MRI image."
    )
    parser.add_argument(
        "-i",
        "--image-path",
        type=Path,
        required=True,
        help="Path to the input MRI image file.",
    )
    parser.add_argument(
        "-m",
        "--mask-path",
        type=Path,
        required=True,
        help="Path to the output mask file.",
    )
    parser.add_argument(
        "-o",
        "--masked-image-path",
        type=Path,
        default=None,
        help="Path to the masked image file.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if is_cuda_available() else "cpu",
        help="Device to run the ONNX model on (default: cpu).",
    )
    args = parser.parse_args()

    predict(args.image_path, args.mask_path, args.masked_image_path, args.device)
