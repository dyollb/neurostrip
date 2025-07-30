# Contributing to NeuroStrip

We welcome contributions! This guide will help you get started with development.

## Quick Start

1. Fork and clone the repository
2. Set up development environment: `make setup-dev`
3. Make your changes
4. Run quality checks: `make check`
5. Submit a pull request

## Development Environment

### Setup
```bash
make setup-dev
```
This installs the package in editable mode with development dependencies and sets up pre-commit hooks.

### Tools
- **Ruff**: Linting and formatting
- **MyPy**: Type checking
- **Bandit**: Security scanning
- **pytest**: Testing
- **pre-commit**: Automated quality checks
- **Safety**: Dependency vulnerability scanning (dev dependency only)
- **Typos**: Spell checking

## Available Commands

```bash
make help              # Show all commands
make format            # Format code with ruff
make lint              # Run ruff linter
make fix               # Auto-fix linting issues
make type-check        # Run mypy type checker (excludes tests)
make security          # Security scan with bandit
make test              # Run tests with pytest
make test-cov          # Tests with coverage report
make check             # Run all checks (format, lint, type-check, test)
make pre-commit-run    # Run pre-commit on all files
make clean             # Clean build artifacts
```

## Code Standards

### Quality Requirements
- **Type annotations**: Required for all public functions (MyPy excludes tests)
- **Test coverage**: Must maintain or improve existing coverage
- **Pre-commit hooks**: Must pass all automated checks
- **Python 3.9+**: Modern Python features encouraged
- **Ruff compliance**: Code must pass linting and formatting checks

### Pre-commit Hooks
Automatically run on commit:
- General checks: trailing whitespace, end-of-file fixing, YAML/TOML validation, large files, merge conflicts
- Code formatting and linting (Ruff)
- Type checking (MyPy, excludes tests)
- Security scanning (Bandit, excludes tests)
- Spell checking (Typos)

## Testing

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
python -m pytest tests/test_main.py -v

# Specific test function
python -m pytest tests/test_main.py::test_predict_basic -v
```

## Workflow

1. Create feature branch: `git checkout -b feature-name`
2. Make changes in `src/` directory
3. Run `make check` before committing
4. Commit (hooks run automatically)
5. Push and create pull request

## Dependencies

- **Runtime**: Add to `dependencies` in `pyproject.toml`
- **Development**: Add to `dev` optional dependencies
- **GPU support**: Use `gpu` optional dependencies

## Project Structure

```
src/neurostrip/          # Main package
tests/                   # Test files
pyproject.toml          # Project config
.pre-commit-config.yaml # Hook config
Makefile               # Dev commands
```

## Pull Request Guidelines

All contributions must:
- Pass pre-commit hooks
- Maintain test coverage
- Include type annotations
- Pass CI checks

Questions? Open an issue or start a discussion.
