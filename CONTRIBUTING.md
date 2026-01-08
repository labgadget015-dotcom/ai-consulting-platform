# Contributing to AI Consulting Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to:
- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- PostgreSQL (for local development)
- Virtual environment tool (venv, conda, etc.)

### Setup Development Environment

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/ai-consulting-platform.git
cd ai-consulting-platform
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

4. **Install pre-commit hooks**

```bash
pre-commit install
```

5. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your local configuration
```

6. **Run tests to verify setup**

```bash
pytest
```

## Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Urgent production fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Making Changes

1. **Create a new branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Write clear, concise commit messages
   - Follow the coding standards below
   - Add tests for new functionality

3. **Run pre-commit checks**

```bash
pre-commit run --all-files
```

4. **Commit your changes**

```bash
git add .
git commit -m "feat: add new feature description"
```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Example:
```
feat: add inventory optimization algorithm

Implement ML-based inventory optimization using Prophet
for demand forecasting with 95% confidence intervals.

Closes #123
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 88)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints for function signatures

### Code Quality Tools

All code must pass:

- **Black** - Code formatting
- **Flake8** - Linting (max line length: 88)
- **Pylint** - Static code analysis (minimum score: 8.0)
- **Bandit** - Security scanning (no high/medium severity issues)
- **mypy** - Type checking

### Documentation

- Add docstrings to all public modules, functions, classes, and methods
- Use [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings
- Update README.md if adding new features
- Add inline comments for complex logic

Example:

```python
def calculate_forecast(data: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Generate sales forecast using Prophet algorithm.
    
    Args:
        data: Historical sales data with 'ds' and 'y' columns
        periods: Number of days to forecast (default: 30)
    
    Returns:
        DataFrame containing forecast with confidence intervals
    
    Raises:
        ValueError: If data is empty or missing required columns
    """
    pass
```

## Testing

### Test Requirements

- All new features must include tests
- Maintain minimum 80% code coverage
- Tests must pass before PR approval

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_forecasting.py

# Run tests matching pattern
pytest -k "test_inventory"
```

### Writing Tests

- Use `pytest` framework
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use fixtures for common setup

Example:

```python
import pytest
from app.forecasting import calculate_forecast

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=100),
        'y': np.random.randn(100).cumsum()
    })

def test_forecast_returns_correct_shape(sample_data):
    result = calculate_forecast(sample_data, periods=30)
    assert len(result) == 30
    assert 'yhat' in result.columns
```

## Pull Request Process

### Before Submitting

1. ✅ All tests pass locally
2. ✅ Code coverage meets minimum threshold (80%)
3. ✅ Pre-commit hooks pass
4. ✅ Documentation is updated
5. ✅ Commit messages follow convention
6. ✅ Branch is up to date with main

### Submitting a PR

1. **Push your branch**

```bash
git push origin feature/your-feature-name
```

2. **Create Pull Request on GitHub**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Reference related issues (e.g., "Closes #123")
   - Add screenshots/GIFs for UI changes

3. **PR Review Process**
   - Wait for CI/CD checks to pass
   - Address reviewer feedback promptly
   - Keep the PR focused and reasonably sized
   - Respond to comments and questions

4. **After Approval**
   - Squash commits if requested
   - Maintainer will merge the PR

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated

## Related Issues
Closes #(issue number)

## Screenshots (if applicable)
```

## Questions?

Feel free to:
- Open an issue for bugs or feature requests
- Start a discussion for questions
- Reach out to maintainers

Thank you for contributing! 🎉
