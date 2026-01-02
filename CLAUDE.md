# Coding Practices for Python Functions
- Always add type hinting in the function arguments and for the return type
- Ensure each function is less than 24 lines long and less than 80 characters wide
- Store functions in a `./src` folder that you create and treat as a local module
- Use `uv` unless the instructions or documentation specify otherwise

# Coding Practices for Python Classes
- Avoid classes in Python. Prioritize functions. Only use classes if it is absolutely necessary.

# Coding Practices when Python Files are Modified
- Always use `ruff format` to format files when you make changes, unless documentation or instructions specify otherwise
- Always use `ruff check --fix` to lint and fix python files

# Running Python Files
- Use the following tool call for modal:
```
Bash(export PYTHONIOENCODING="utf-8"; uv run modal run modal_quickstart.py)
```
