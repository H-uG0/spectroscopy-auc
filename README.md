# Spectroscopy AUC Extraction Pipeline

[![Python application](https://github.com/hugo/spectroscopy-auc/actions/workflows/python-app.yml/badge.svg)](https://github.com/hugo/spectroscopy-auc/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

A robust, reusable Python pipeline and Streamlit application for analyzing spectroscopy data and extracting the Area Under the Curve (AUC).

This tool allows you to easily compute the integrated area between reference and sample spectra across multiple image files, complete with spline interpolation and data visualization.

## Features

- **Automated Processing**: Process multiple spectroscopy `.tif` images in a batch.
- **Robust Algorithms**: Extracts Reference and Sample profiles, aligns backgrounds, and calculates AUC using standard numerical integration or spline-based curves.
- **Interactive UI**: A Streamlit web application to visualize the spectral curves, analyze differences, and tune processing parameters visually.
- **Flexible Data Export**: Outputs results to easily consumable CSV files.
- **Simulations**: Includes scripts to generate test datasets for validation.

## Prerequisites

- Python 3.9 or higher.

## Installation

It is recommended to use a virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/hugo/spectroscopy-auc.git
cd spectroscopy-auc

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate

# 3. Install the application and dependencies
pip install -e .
```

To install development dependencies (for testing and formatting):

```bash
pip install -e .[dev]
```

## Usage

### 1. Web Interface (Streamlit)

The easiest way to use the tool is via the interactive Streamlit dashboard.

```bash
python -m streamlit run src/ui/app.py
```

This will open a browser window where you can:
- Select a directory or upload a `.json` configuration file.
- View the reference and sample spectral profiles.
- Visualize the calculated area over the background.
- Adjust axis scaling and units.

### 2. Programmatic Usage

You can also import the underlying modules into your own Python scripts:

```python
from src.processing.core import process_single_file
from src.processing.visualization import calculate_auc

# Example usage (refer to src/main.py for full context)
result = process_single_file("path/to/image.tif")
print(result)
```

## Running Tests

To ensure everything is working correctly:

```bash
pytest
```

## Code Quality

This project uses `black`, `isort`, and `flake8` for formatting and linting.

```bash
# Format code
black .
isort .

# Run static analysis
flake8 .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Make sure to update tests as appropriate and ensure that the CI pipeline passes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
