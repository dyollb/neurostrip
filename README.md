# NeuroStrip
[![CI](https://github.com/dyollb/neurostrip/actions/workflows/ci.yml/badge.svg)](https://github.com/dyollb/neurostrip/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/neurostrip.svg)](https://pypi.org/project/neurostrip/)
<img src="https://img.shields.io/pypi/dm/neurostrip.svg?label=pypi%20downloads&logo=python&logoColor=green"/>
<img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776ab.svg"/>

CNN based skull stripping (brain masking) from MRI

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
neurostrip --image-path input.nii.gz --output-path output.nii.gz
```

## Development

### Setup Development Environment

1. Clone the repository
2. Set up the development environment:
   ```bash
   make setup-dev
   ```

This will:
- Install the package in editable mode with development dependencies
- Install pre-commit hooks for automatic code quality checks

### Development Tools

The project uses modern Python development tools optimized for Python ≥3.10:

- **Ruff**: Ultra-fast Python linter and formatter (replaces Black, isort, flake8, and more)
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning
- **pytest**: Testing framework
- **pre-commit**: Git hooks for automated code quality

### Available Commands

```bash
make help              # Show all available commands
make format            # Format code with ruff
make lint              # Run ruff linter
make fix               # Run ruff linter with auto-fix
make type-check        # Run mypy type checker
make security          # Run bandit security scanner
make test              # Run tests with pytest
make test-cov          # Run tests with coverage report
make check             # Run all checks (format, lint, type-check, test)
make pre-commit-run    # Run pre-commit on all files
make clean             # Clean build artifacts
```

### Pre-commit Hooks

Pre-commit hooks run automatically on every commit to ensure code quality:

- **General checks**: trailing whitespace, end-of-file fixing, YAML/TOML validation
- **Ruff linting and formatting**: comprehensive Python code analysis and formatting
- **MyPy type checking**: static type analysis
- **Bandit security scanning**: security vulnerability detection
- **Safety dependency checking**: known security vulnerabilities in dependencies
- **Typos spell checking**: documentation and comment spell checking

### Code Quality Standards

The project enforces strict code quality standards:

#### Ruff Configuration
- **Target**: Python 3.10+
- **Rules**: Comprehensive set including pycodestyle, pyflakes, isort, pyupgrade, simplify, type-checking, and pylint

#### Type Checking
- **MyPy**: Strict type checking enabled
- **Required**: Type annotations for all public functions
- **Configuration**: Comprehensive warnings and strict equality checks

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
python -m pytest tests/test_main.py -v

# Run specific test function
python -m pytest tests/test_main.py::test_predict_basic -v
```

### Development Workflow

1. **Make changes**: Edit code in `src/` directory
2. **Format and check**: Run `make check` to ensure code quality
3. **Test**: Run `make test` to verify functionality
4. **Commit**: Pre-commit hooks will run automatically
5. **Push**: Create pull request

### Adding Dependencies

- **Runtime dependencies**: Add to `dependencies` in `pyproject.toml`
- **Development dependencies**: Add to `dev` optional dependencies
- **GPU support**: Already configured with `gpu` optional dependencies

### Project Structure

```
neurostrip/
├── src/neurostrip/          # Main package source
├── tests/                   # Test files
├── pyproject.toml          # Project configuration
├── .pre-commit-config.yaml # Pre-commit hook configuration
├── Makefile               # Development commands
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Set up development environment: `make setup-dev`
4. Make your changes
5. Run quality checks: `make check`
6. Commit changes (pre-commit hooks will run automatically)
7. Push and create a pull request

All contributions must:
- Pass pre-commit hooks
- Maintain or improve test coverage
- Include type annotations
- Pass all CI checks

## License

MIT License - see LICENSE file for details.
