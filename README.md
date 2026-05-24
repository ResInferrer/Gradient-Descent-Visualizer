# Gradient Descent Visualizer: C++ Core + Python GUI

![Python Version](https://img.shields.io/badge/python-3.13%252B-blue.svg) ![C++](https://img.shields.io/badge/C%2B%2B-17%2B-00599C.svg?logo=c%2B%2B) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-orange.svg) ![License](https://img.shields.io/badge/license-MIT-yellow.svg) 

TODO: fill out the readme

## **Overview**  
- Visualization of logistic regression and SVM.
- The computational core (loss calculation, gradients, and weight updates) is written in **C++** and wrapped as a Python module using **pybind11**.  
- The user interface and graphics are implemented in **Python** (`matplotlib`, `numpy`).  
- The application runs inside a **Docker** container, ensuring a fully reproducible environment.  

## **Quick Start**
**Clone and set up the project:**
```
git clone https://github.com/KotingGG/Gradient-Descent-Visualizer.git
cd Gradient-Descent-Visualizer
```

**Install dependencies:**
```
# Windows
py -m venv venv
venv\Scripts\activate 

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

**Install the project in development mode:**
```
pip install -e .
```

**Launch an entry point:**
```
python main.py
```

TODO: docker!

## **Tests**
**Launch a test python:**
```
python -m tests.test_core
```

## **Folder structure:**
```
Gradient-Descent-Visualizer/
├── core/                                 # C++ computational core
│   ├── bindings/                         
│   │   ├── CMakeLists.txt
│   │   └── pybind_wrapper.cpp            # Export classes and functions to Python
│   ├── include/                          
│   │   └── gdv/                          
│   │       ├── losses/
│   │       │   ├── BinaryCrossEntropy.h
│   │       │   ├── HingeLoss.h
│   │       │   ├── ILossFunction.h
│   │       │   └── MSELoss.h
│   │       ├── models/
│   │       │   ├── LogisticRegression.h
│   │       │   └── SVM.h
│   │       └── optimizers 
│   │           └── Optimizers.h     
│   ├── src/                              # C++ module implementations
│   │   ├── losses/
│   │   │   ├── BinaryCrossEntropy.cpp
│   │   │   ├── HingeLoss.cpp
│   │   │   └── MSELoss.cpp
│   │   ├── models/
│   │   │   ├── LogisticRegression.cpp
│   │   │   └── SVM.cpp
│   │   └── optimizers 
│   │       └── Optimizers.cpp
│   └── CMakeLists.txt                    # Build C++ library and Python module
│
├── python/                               
│   └── gdv_vis/                          
│       ├── __init__.py
│       ├── data_generator.py            
│       ├── visualizer.py                 
│       └── runner.py                     # Glue: calls C++ module, dataset generation and visualization
│
├── tests/
│   └── test_core.py                      # Test calling a module from Python
│
├── graph/                                # Visualization of models: metrics, features on a graph, etc.
│   ├── logistic_regression_binary.png
│   ├── logistic_regression_multiclass.png
│   └── svm_binary.png
│
├── docs/
│   ├── architecture.md                   # Architectural overview (C++/Python integration)
│   └── usage.md                          # Setup and launch instructions
│
├── main.py                               # Application entry point
├── CMakeLists.txt                        # Root CMake (manages entire C++ build)
├── Dockerfile                            
├── requirements.txt                      
├── pyproject.toml                        
├── .gitignore
└── README.md
```

