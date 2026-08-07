## Contributing

All kinds of contributions are welcome, including but not limited to the following.

- Fix typo or bugs
- Add documentation or translate the documentation into other languages
- Add new features and components

### Workflow

1. fork and pull the latest repository
2. checkout a new branch (do not commit changes directly to the main branch)
3. commit your changes
4. create a PR

Note: If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.

### Code style

#### Python

Install the repository's checks before making a pull request:

```bash
pre-commit install
pre-commit run --all-files
```

The checks use Ruff for Python linting and formatting, docformatter for
docstrings, mdformat for Markdown, and clang-format for C++.
