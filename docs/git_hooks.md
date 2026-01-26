# Triggering CI/CD via Git Hooks

To ensure code quality before pushing to the repository, you can set up Git hooks to run the same checks as the CI pipeline.

## 1. Using `pre-commit` Framework (Recommended)

The `pre-commit` framework manages git hooks for you.

### Installation
```bash
pip install pre-commit
```

### Configuration (.pre-commit-config.yaml)
Create a `.pre-commit-config.yaml` file in the root directory:

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-requests]

-   repo: local
    hooks:
    -   id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
    -   id: validate-model
        name: Validate Model
        entry: python scripts/validate_model.py
        language: system
        pass_filenames: false
```

### Activation
Run correct command to install the hooks:
```bash
pre-commit install
```

Now, every time you commit, these checks will run automatically.

## 2. Using Standard Git Hooks (Manual)

If you prefer not to use `pre-commit`, you can create a simple shell script in `.git/hooks/pre-push`.

1.  Create file `.git/hooks/pre-push`
2.  Add content:

```bash
#!/bin/sh

echo "Running Pre-Push Checks..."

# 1. Linting
echo "Running Flake8..."
flake8 . || exit 1

# 2. Testing
echo "Running Tests..."
pytest || exit 1

# 3. Model Validation
echo "Validating Model..."
python scripts/validate_model.py || exit 1

echo "All checks passed!"
exit 0
```

3.  Make it executable (Linux/Mac) or verify widely on Windows (Git Bash).
