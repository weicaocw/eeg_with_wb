# My Pipeline Project

This is a modular pipeline-based project.

## Structure

- `/config`: Stores `.conf` configuration files.
- `/data`: Stores raw data.
- `/results`: Stores all output results.
- `/src/my_pipeline`: All Python source code packages.
- `/tests`: All unit and integration tests.
- `main.py`: Main entry point of the project.

## Installation

1.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  Install the project in editable mode:
    ```bash
    pip install -e .
    ```

3.  Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Run

To run the project:
```bash
python main.py --config config/default.conf
```

To run tests:
```bash
pytest
```
