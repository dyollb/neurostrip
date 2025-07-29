"""Test the lib module functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from neurostrip.lib import (
    URL,
    ONNXPredictor,
    download_file,
    get_onnxruntime_session,
    is_cuda_available,
    predict,
    sliding_window_inference,
)


class MockPredictor:
    """Mock predictor for testing sliding window inference."""

    def __init__(self, output_channels: int = 2):
        self.output_channels = output_channels

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        """Mock prediction that returns a simple pattern based on input shape."""
        batch_size = input_array.shape[0]
        spatial_dims = input_array.shape[2:]  # Skip batch and channel dims

        # Create output with the specified number of channels
        output_shape = (batch_size, self.output_channels, *spatial_dims)
        output = np.zeros(output_shape, dtype=np.float32)

        # Fill with a simple pattern for testing
        for b in range(batch_size):
            for c in range(self.output_channels):
                # Create a pattern based on channel and batch index
                output[b, c] = c + b * 0.1

        return output


def test_sliding_window_inference_basic():
    """Test basic sliding window inference functionality."""
    # Create a simple 3D input with batch and channel dimensions
    img = np.ones((1, 1, 64, 64, 64), dtype=np.float32)
    roi_size = (32, 32, 32)
    sw_batch_size = 2
    predictor = MockPredictor(output_channels=2)
    overlap = 0.25

    result = sliding_window_inference(
        img=img,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
    )

    # Check output shape
    expected_shape = (1, 2, 64, 64, 64)  # Same spatial dims, 2 output channels
    assert result.shape == expected_shape

    # Check that result is not all zeros (predictor should have produced output)
    assert not np.allclose(result, 0)


def test_sliding_window_inference_exact_fit():
    """Test sliding window inference when ROI size exactly fits the image."""
    # Create input where ROI size exactly matches spatial dimensions
    img = np.ones((1, 1, 32, 32, 32), dtype=np.float32)
    roi_size = (32, 32, 32)
    sw_batch_size = 1
    predictor = MockPredictor(output_channels=3)

    result = sliding_window_inference(
        img=img,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=0.0,
    )

    # Should have same spatial dimensions, 3 output channels
    expected_shape = (1, 3, 32, 32, 32)
    assert result.shape == expected_shape

    # With exact fit and no overlap, should have uniform values per channel
    for c in range(3):
        channel_values = result[0, c]
        assert np.allclose(channel_values, channel_values.flat[0])


def test_sliding_window_inference_different_overlaps():
    """Test sliding window inference with different overlap values."""
    img = np.ones((1, 1, 48, 48, 48), dtype=np.float32)
    roi_size = (24, 24, 24)
    sw_batch_size = 2
    predictor = MockPredictor(output_channels=2)

    # Test different overlap values
    for overlap in [0.0, 0.25, 0.5]:
        result = sliding_window_inference(
            img=img,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=overlap,
        )

        assert result.shape == (1, 2, 48, 48, 48)
        assert not np.allclose(result, 0)


def test_sliding_window_inference_batch_size_variations():
    """Test sliding window inference with different batch sizes."""
    img = np.ones((1, 1, 40, 40, 40), dtype=np.float32)
    roi_size = (20, 20, 20)
    predictor = MockPredictor(output_channels=2)
    overlap = 0.25

    # Test different batch sizes
    for sw_batch_size in [1, 2, 4, 8]:
        result = sliding_window_inference(
            img=img,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=overlap,
        )

        assert result.shape == (1, 2, 40, 40, 40)
        assert not np.allclose(result, 0)


def test_sliding_window_inference_non_cubic_roi():
    """Test sliding window inference with non-cubic ROI sizes."""
    img = np.ones((1, 1, 60, 80, 40), dtype=np.float32)
    roi_size = (30, 40, 20)  # Non-cubic ROI
    sw_batch_size = 2
    predictor = MockPredictor(output_channels=2)
    overlap = 0.25

    result = sliding_window_inference(
        img=img,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
    )

    assert result.shape == (1, 2, 60, 80, 40)
    assert not np.allclose(result, 0)


def test_sliding_window_inference_edge_cases():
    """Test sliding window inference edge cases."""
    # Test with minimum possible image size
    img = np.ones((1, 1, 16, 16, 16), dtype=np.float32)
    roi_size = (16, 16, 16)
    sw_batch_size = 1
    predictor = MockPredictor(output_channels=1)

    result = sliding_window_inference(
        img=img,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=0.0,
    )

    assert result.shape == (1, 1, 16, 16, 16)


def test_sliding_window_inference_preserves_dtype():
    """Test that sliding window inference preserves input dtype."""
    for dtype in [np.float32, np.float64]:
        img = np.ones((1, 1, 32, 32, 32), dtype=dtype)
        roi_size = (16, 16, 16)
        sw_batch_size = 2
        predictor = MockPredictor(output_channels=2)

        result = sliding_window_inference(
            img=img,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=0.25,
        )

        # Result dtype should match input dtype
        assert result.dtype == dtype


class TestONNXPredictor:
    """Test the ONNXPredictor class."""

    @patch("neurostrip.lib.get_onnxruntime_session")
    def test_init(self, mock_get_session):
        """Test ONNXPredictor initialization."""
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_get_session.return_value = mock_session

        model_path = Path("dummy_model.onnx")
        predictor = ONNXPredictor(model_path, device="cpu")

        assert predictor.input_name == "input"
        assert predictor.output_name == "output"
        mock_get_session.assert_called_once_with(model_path, device="cpu")

    @patch("neurostrip.lib.get_onnxruntime_session")
    def test_call_with_numpy_array(self, mock_get_session):
        """Test ONNXPredictor call with numpy array."""
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]

        # Mock the session.run method
        expected_output = np.array([[1, 2, 3]])
        mock_session.run.return_value = [expected_output]
        mock_get_session.return_value = mock_session

        predictor = ONNXPredictor(Path("dummy_model.onnx"))
        input_array = np.array([[4, 5, 6]])

        result = predictor(input_array)

        assert np.array_equal(result, expected_output)
        mock_session.run.assert_called_once_with(["output"], {"input": input_array})

    @patch("neurostrip.lib.get_onnxruntime_session")
    def test_call_with_invalid_input(self, mock_get_session):
        """Test ONNXPredictor call with invalid input type."""
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_get_session.return_value = mock_session

        predictor = ONNXPredictor(Path("dummy_model.onnx"))

        with pytest.raises(TypeError, match="Input must be a NumPy array"):
            predictor([1, 2, 3])  # List instead of numpy array


def test_is_cuda_available():
    """Test CUDA availability detection."""
    result = is_cuda_available()
    assert isinstance(result, bool)


@patch("neurostrip.lib.ort.InferenceSession")
@patch("neurostrip.lib.is_cuda_available")
def test_get_onnxruntime_session_cuda(mock_is_cuda, mock_inference_session):
    """Test getting ONNX runtime session with CUDA."""
    mock_is_cuda.return_value = True
    mock_session = Mock()
    mock_inference_session.return_value = mock_session

    model_path = Path("dummy_model.onnx")
    session = get_onnxruntime_session(model_path, device="cuda")

    assert session == mock_session
    mock_inference_session.assert_called_once_with(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )


@patch("neurostrip.lib.ort.InferenceSession")
@patch("neurostrip.lib.is_cuda_available")
def test_get_onnxruntime_session_cuda_unavailable(mock_is_cuda, mock_inference_session):
    """Test getting ONNX runtime session when CUDA is unavailable."""
    mock_is_cuda.return_value = False

    model_path = Path("dummy_model.onnx")

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is not available"):
        get_onnxruntime_session(model_path, device="cuda")


@patch("neurostrip.lib.ort.InferenceSession")
def test_get_onnxruntime_session_cpu(mock_inference_session):
    """Test getting ONNX runtime session with CPU."""
    mock_session = Mock()
    mock_inference_session.return_value = mock_session

    model_path = Path("dummy_model.onnx")
    session = get_onnxruntime_session(model_path, device="cpu")

    assert session == mock_session
    mock_inference_session.assert_called_once_with(
        model_path, providers=["CPUExecutionProvider"]
    )


@patch("neurostrip.lib.sitk.ReadImage")
@patch("neurostrip.lib.sitk.WriteImage")
def test_predict_basic(mock_write: Mock, mock_read: Mock) -> None:
    """Test basic predict functionality."""
    # Mock the image
    mock_image = Mock()
    mock_image.GetSpacing.return_value = [1.0, 1.0, 1.0]
    mock_image.GetSize.return_value = [256, 256, 256]
    mock_read.return_value = mock_image

    mock_array = np.ones((256, 256, 256), dtype=np.float32)

    # Mock the sliding window inference output
    mock_inference_output = np.ones((1, 2, 256, 256, 256), dtype=np.float32)

    # Mock the output of DICOMOrient
    with (
        patch("neurostrip.lib.sitk.DICOMOrient", return_value=mock_image),
        patch("neurostrip.lib.ONNXPredictor"),
        patch(
            "neurostrip.lib.sliding_window_inference",
            return_value=mock_inference_output,
        ),
        patch("neurostrip.lib.sitk.GetArrayFromImage", return_value=mock_array),
        patch("neurostrip.lib.sitk.GetImageFromArray", return_value=mock_image),
        patch("neurostrip.lib.np.transpose", return_value=mock_array),
        patch("neurostrip.lib.np.expand_dims", return_value=mock_array),
        patch("neurostrip.lib.np.argmax", return_value=mock_array.astype(np.uint8)),
    ):
        predict(
            image_path=Path("dummy_input.nii"),
            mask_path=Path("dummy_output.nii"),
            device="cpu",
        )

    # Verify that WriteImage was called
    mock_write.assert_called_once()


def test_download_file_success():
    """Test successful download and extraction of the model file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dest_folder = Path(temp_dir)

        # Download and extract the file
        result = download_file(URL, dest_folder, timeout=30.0)

        # Assert download was successful
        assert result is True

        # Assert the expected model file exists after extraction
        model_file = dest_folder / "brainmask.onnx"
        assert model_file.exists()
        assert model_file.stat().st_size > 0

        # Assert zip file was cleaned up
        zip_file = dest_folder / "brainmask.onnx-1.0.zip"
        assert not zip_file.exists()
