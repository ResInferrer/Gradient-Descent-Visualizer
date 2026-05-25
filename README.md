# Gradient Descent Visualizer

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17%2B-00599C.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**Gradient Descent Visualizer** is an educational tool that demonstrates how logistic regression and SVM work.  
The computationally intensive parts (loss calculation, gradient computation, weight updates) are implemented in **C++** and exposed to Python via **pybind11**.  
The frontend uses **Python** (`matplotlib`, `scikit-learn`, `numpy`) for data generation, metrics, and visualisation.

The project can be run natively on Linux/macOS/Windows or inside a **Docker** container for a completely reproducible environment.

---

## Table of Contents

- [Gradient Descent Visualizer](#gradient-descent-visualizer)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Local Installation (without Docker)](#local-installation-without-docker)
    - [Linux / macOS](#linux--macos)
    - [Windows](#windows)
  - [Running the Application Locally](#running-the-application-locally)
  - [Running with Docker](#running-with-docker)
    - [Build the Docker image](#build-the-docker-image)
    - [Run the container interactively](#run-the-container-interactively)
  - [Running Tests](#running-tests)
  - [Project Structure](#project-structure)
  - [Notes](#notes)
  - [License](#license)

---

## Requirements

- **C++ compiler** with C++17 support (g++ 9+, clang 10+, or MSVC 2019+)
- **CMake** (>= 3.15)
- **Python** (>= 3.11)
- **pip** and **virtualenv** (recommended)

For Docker:
- **Docker Engine** (or Docker Desktop) installed and running.

---

## Local Installation (without Docker)

These steps build the C++ module and install Python dependencies directly on your host system.

### Linux / macOS

```bash
# Clone the repository
git clone https://github.com/ResInferrer/Gradient-Descent-Visualizer.git
cd Gradient-Descent-Visualizer

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Build the C++ extension module
mkdir build && cd build
cmake ..
make

# Copy the generated shared library to the project root
cp gdv_vis.so ..

# Return to project root
cd ..
```

### Windows

```powershell
git clone https://github.com/ResInferrer/Gradient-Descent-Visualizer.git
cd Gradient-Descent-Visualizer

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Build the C++ module (adjust generator if needed)
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# Copy the .pyd file (Windows equivalent of .so) to the project root
copy Release\gdv_vis.pyd ..

cd ..
```

---

## Running the Application Locally

After a successful local build, launch the visualiser:

```bash
python main.py
```

You will be prompted to choose:
- Model: `0` for Logistic Regression, `1` for SVM
- For Logistic Regression: `0` for binary classification, `1` for multiclass (One‑vs‑All)

The program will then:
- Generate a synthetic dataset
- Train the selected model
- Display performance metrics and visualisations (PCA projections, confusion matrix, bar charts)

All generated plots are saved in the `graphs/` directory.

---

## Running with Docker

Using Docker eliminates the need to install a C++ toolchain and Python dependencies on your host. It also ensures exactly the same environment across all systems.

### Build the Docker image

```bash
docker build -t gdv:v1 .
```

### Run the container interactively

You **must** use the `-it` flags to enable interactive input.

```bash
docker run --rm -it gdv:v1
```

> `--rm` automatically removes the container after exit.  
> `-it` keeps stdin open and allocates a pseudo‑TTY, which is required for the `input()` prompts.

If you prefer to pre‑select options (e.g., for automation), you can modify `runner.py` to read environment variables; the current version relies on manual input.

---

## Running Tests

Unit tests verify that the C++ module loads correctly and basic functions work.  
**Run them locally (without Docker) after installation.**

```bash
# Activate the virtual environment first (if not already active)
source venv/bin/activate      # Linux/macOS
.\venv\Scripts\activate       # Windows

# Run the test suite
python -m tests.test_core
```

Expected output: a series of OK/FAIL messages. All tests should pass if the C++ module was built correctly.

---

## Project Structure

```
Gradient-Descent-Visualizer/
├── core/                    # C++ computational core
│   ├── bindings/            # pybind11 wrappers
│   ├── include/gdv/         # headers (losses, models, optimizers)
│   ├── src/                 # implementations
│   └── CMakeLists.txt
├── python/gdv_vis/          # Python frontend
│   ├── data_generator.py    # dataset generation (scikit-learn)
│   ├── visualizer.py        # plotting functions (matplotlib, seaborn)
│   ├── runner.py            # main controller (C++ model calls, metrics)
│   └── __init__.py
├── tests/
│   └── test_core.py         # basic module import test
├── graphs/                  # saved plots (created at runtime)
├── main.py                  # entry point
├── requirements.txt         # Python dependencies
├── Dockerfile               # container definition
├── CMakeLists.txt           # root CMake configuration
├── pyproject.toml
└── README.md
```

---

## Notes

- The Docker approach is **optional** but recommended for grading or if you want a completely reproducible environment. It is not overkill – it guarantees that your C++ module will be compiled with the exact same library versions as the reviewer’s system.
- All visualisations are saved in the `graphs/` folder; they are not displayed in the Docker output (since no GUI is available). To see them, either run locally or copy the generated PNG files from the container using `docker cp`.
- On Windows, if you use Docker Desktop with Linux containers, the build inside Docker is identical to the Linux instructions. The local build requires a C++ compiler (MinGW or Visual Studio) and CMake.

---

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.