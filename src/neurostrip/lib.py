import typing
from math import ceil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import SimpleITK as sitk


def predict(
    image_path: Path,
    mask_path: Path,
    masked_image_path: Path | None = None,
    device: str = "cuda",
) -> None:
    """Predict brain mask from a 3D MRI image and save the output mask."""
    input = sitk.ReadImage(image_path, sitk.sitkFloat32)
    img = sitk.DICOMOrient(input, "RAS")  # Ensure image is in RAS orientation
    dx = 1.0  # Desired spacing in mm
    tolerance = 0.1  # Allowable spacing deviation
    if any(abs(s - dx) > tolerance for s in img.GetSpacing()):
        size = [ceil(n * s / dx) for n, s in zip(img.GetSize(), img.GetSpacing())]
        img = sitk.Resample(
            img,
            size,
            sitk.Transform(),
            sitk.sitkLinear,
            img.GetOrigin(),
            [dx] * 3,
            img.GetDirection(),
            0.0,
            img.GetPixelID(),
        )
    img_np = sitk.GetArrayFromImage(img)
    # Transpose from (z, y, x) to (x, y, z) for compatibility with MONAI/NumPy
    img_np = np.transpose(img_np, (2, 1, 0))
    # Normalize image
    mean, sdev = img_np.mean(), img_np.std()
    img_np = (img_np - mean) / sdev if sdev > 0 else img_np - mean

    # Load ONNX model
    model_path = (
        Path("/Users/lloyd/Dropbox/Work/Data/ML_models/BrainMask") / "unet.onnx"
    )
    predictor = ONNXPredictor(model_path, device=device)

    # Run inference
    # Add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
    img_np = np.expand_dims(np.expand_dims(img_np, 0), 0)
    mask_np = sliding_window_inference(
        img_np,
        roi_size=(96, 96, 96),
        sw_batch_size=4,
        predictor=predictor,
        overlap=0.25,
    )
    # Remove batch and channel dimensions and create SimpleITK image
    # - Use argmax to get the predicted class
    mask_np = np.argmax(mask_np[0], axis=0).astype(np.uint8)
    # Transpose back to (z, y, x) for SimpleITK
    mask_np = np.transpose(mask_np, (2, 1, 0))
    mask = sitk.GetImageFromArray(mask_np)
    mask.CopyInformation(img)
    if mask.GetSize() != input.GetSize():
        mask = sitk.Resample(mask, input, sitk.Transform(), sitk.sitkLabelLinear)
    sitk.WriteImage(mask, mask_path)

    if masked_image_path:
        input = sitk.Mask(input, mask)
        sitk.WriteImage(input, masked_image_path)


class ONNXPredictor:
    def __init__(self, model_path: Path, device: str = "cuda"):
        """Initialize the ONNX predictor with the model path and device."""
        self.session = get_onnxruntime_session(model_path, device=device)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        if not isinstance(input_array, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        output = self.session.run([self.output_name], {self.input_name: input_array})[0]
        return typing.cast(np.ndarray, output)


def is_cuda_available() -> bool:
    """Check if CUDA is available for ONNX Runtime."""
    return "CUDAExecutionProvider" in ort.get_available_providers()


def get_onnxruntime_session(
    model_path: Path, device: str = "cuda"
) -> ort.InferenceSession:
    if device == "cuda":
        if is_cuda_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. Please check your ONNX Runtime installation."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def sliding_window_inference(
    img: np.ndarray,
    roi_size: tuple[int, ...],
    sw_batch_size: int,
    predictor: ONNXPredictor,
    overlap: float = 0.25,
) -> np.ndarray:
    """Pure NumPy version of sliding window inference."""
    ndim = img.ndim - 2
    strides = [int(r * (1 - overlap)) for r in roi_size]
    img_shape = img.shape[2:]

    # Calculate slices
    slices = []
    for i in range(ndim):
        s = list(range(0, img_shape[i] - roi_size[i] + 1, strides[i]))
        if s[-1] + roi_size[i] < img_shape[i]:
            s.append(img_shape[i] - roi_size[i])
        slices.append(s)

    window_locations = [
        tuple(coord)
        for coord in np.stack(np.meshgrid(*slices, indexing="ij"), -1).reshape(-1, ndim)
    ]

    # Get the first batch to determine output shape
    first_coord = window_locations[0]
    first_patch_key = (...,) + tuple(
        slice(first_coord[d], first_coord[d] + roi_size[d]) for d in range(ndim)
    )
    first_patch = img[first_patch_key]
    sample_output = predictor(first_patch)
    output_channels = sample_output.shape[1]

    # Initialize output arrays with correct number of channels
    output_shape = (img.shape[0], output_channels, *img.shape[2:])
    output = np.zeros(output_shape, dtype=img.dtype)
    count_map = np.zeros(output_shape, dtype=img.dtype)

    # Process first patch (already computed)
    first_slices = tuple(
        slice(first_coord[d], first_coord[d] + roi_size[d]) for d in range(ndim)
    )
    output_key = (...,) + first_slices
    output[output_key] += sample_output[0]
    count_map[output_key] += 1

    # Process remaining patches
    for idx in range(1, len(window_locations), sw_batch_size):
        batch_coords = window_locations[idx : idx + sw_batch_size]
        batch_patches = []
        for coord in batch_coords:
            patch_key = (...,) + tuple(
                slice(coord[d], coord[d] + roi_size[d]) for d in range(ndim)
            )
            patch = img[patch_key]
            batch_patches.append(patch)
        batch_input = np.concatenate(batch_patches, axis=0)

        batch_output = predictor(batch_input)

        for b, coord in enumerate(batch_coords):
            slice_key = (...,) + tuple(
                slice(coord[d], coord[d] + roi_size[d]) for d in range(ndim)
            )
            output[slice_key] += batch_output[b]
            count_map[slice_key] += 1

    return output / count_map
