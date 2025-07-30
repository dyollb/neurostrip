# NeuroStrip
[![CI](https://github.com/dyollb/neurostrip/actions/workflows/ci.yml/badge.svg)](https://github.com/dyollb/neurostrip/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/neurostrip.svg)](https://pypi.org/project/neurostrip/)
<img src="https://img.shields.io/pypi/dm/neurostrip.svg?label=pypi%20downloads&logo=python&logoColor=green"/>
<img src="https://img.shields.io/badge/python-3.9%20|3.10%20|%203.11%20|%203.12-3776ab.svg"/>

CNN based skull stripping (brain masking) from MRI.

<p align="center">
   <img src="https://raw.githubusercontent.com/dyollb/neurostrip/main/slicer_plugin/Resources/Icons/NeuroStrip.png" alt="NeuroStrip Slicer Plugin Icon" width="120"/>
</p>

## Installation

For CPU support:
```bash
pip install neurostrip[cpu]
```

For GPU support:
```bash
pip install neurostrip[gpu]
```

## Usage

```bash
neurostrip --image-path input.nii.gz --mask-path mask.nii.gz --masked-image-path output.nii.gz
```

## Slicer Plugin

To make it more convenient to use, this repository includes a plugin that can be installed in [3D Slicer](https://www.slicer.org/). For installation instructions and more details, please refer to the [README](https://raw.githubusercontent.com/dyollb/neurostrip/main/slicer_plugin/README.md).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see LICENSE file for details.
