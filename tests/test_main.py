"""Test the main module functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

from neurostrip.main import main


@patch("neurostrip.main.predict")
def test_main_with_required_args(mock_predict: Mock) -> None:
    """Test main function with required arguments."""
    test_args = [
        "neurostrip",
        "--image-path",
        "/path/to/input.nii",
        "--mask-path",
        "/path/to/output.nii",
        "--device",
        "cpu",
    ]

    with patch("sys.argv", test_args):
        main()

    # Verify predict was called with correct arguments
    mock_predict.assert_called_once_with(
        Path("/path/to/input.nii"), Path("/path/to/output.nii"), None, "cpu"
    )
