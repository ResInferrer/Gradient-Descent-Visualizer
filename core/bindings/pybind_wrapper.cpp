#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gdv/models/LogisticRegression.h"
#include "gdv/models/SVM.h"

namespace py = pybind11;

PYBIND11_MODULE(gdv_vis, m) {
    m.doc() = "C++ backend for gradient descent visualizer";

    // Wrap Logistic Regression
    py::class_<gdv::models::LogisticRegression>(m, "LogisticRegression")
        .def(py::init<>())
        .def("fit", &gdv::models::LogisticRegression::fit,
             py::arg("X"), py::arg("y"),
             py::arg("learning_rate") = 0.01,
             py::arg("iterations") = 1000,
             py::arg("l2_lambda") = 0.0,
             py::arg("verbose") = false)
        .def("predict_proba", &gdv::models::LogisticRegression::predict_proba)
        .def("predict", &gdv::models::LogisticRegression::predict,
             py::arg("X_row"), py::arg("threshold") = 0.5);

    // Wrap SVM
    py::class_<gdv::models::SVM>(m, "SVM")
        .def(py::init<>())
        .def("fit", &gdv::models::SVM::fit,
             py::arg("X"), py::arg("y"),
             py::arg("C") = 1.0,
             py::arg("learning_rate") = 0.01,
             py::arg("iterations") = 1000,
             py::arg("verbose") = false)
        .def("decision_function", &gdv::models::SVM::decision_function)
        .def("predict", &gdv::models::SVM::predict);
}