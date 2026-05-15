#include "gdv/models/SVM.h"
#include <cmath>
#include <cassert>
#include <iostream>

namespace gdv::models {

SVM::SVM() 
    : weights_(), bias_(0.0) 
{
}

#pragma region Public API

double SVM::decision_function(const std::vector<double>& X_row) const {
    assert(weights_.size() == X_row.size() && "Feature count mismatch");
    
    double result = bias_;
    for (size_t i = 0; i < weights_.size(); ++i) {
        result += weights_[i] * X_row[i];
    }
    return result;
}

int SVM::predict(const std::vector<double>& X_row) const {
    return (decision_function(X_row) >= 0.0) ? 1 : -1;
}

void SVM::fit(const std::vector<std::vector<double>>& X,
              const std::vector<int>& y,
              double C,
              double learning_rate,
              int iterations,
              bool verbose) {
    size_t n_samples = X.size();
    if (n_samples == 0) return;
    size_t n_features = X[0].size();
    assert(y.size() == n_samples && "Mismatch between X and y sizes");
    

    // Initialize weights to zero
    weights_.assign(n_features, 0.0);
    bias_ = 0.0;
    
    // Auxiliary variables
    std::vector<double> grad_w(n_features);
    double grad_b = 0.0;
    
    for (int iter = 0; iter < iterations; ++iter) {
        double loss = hinge_loss_and_subgradient(X, y, C, grad_w, grad_b);
        
        // Update parameters (gradient descent)
        for (size_t j = 0; j < n_features; ++j) {
            weights_[j] -= learning_rate * grad_w[j];
        }
        bias_ -= learning_rate * grad_b;
    }
}

#pragma endregion

#pragma region Private API

double SVM::hinge_loss_and_subgradient(const std::vector<std::vector<double>>& X,
                                       const std::vector<int>& y,
                                       double C,
                                       std::vector<double>& grad_w,
                                       double& grad_b) const {
    size_t n_samples = X.size();
    size_t n_features = (n_samples == 0) ? 0 : X[0].size();
    
    // Initialize gradients
    grad_w.assign(n_features, 0.0);
    grad_b = 0.0;
    double loss = 0.0;
    
    // Regularization part of losses
    double reg_loss = 0.0;
    for (double w : weights_) {
        reg_loss += w * w;
    }
    reg_loss *= 0.5;
    loss += reg_loss;
    
    // Subgradients from regularization
    for (size_t j = 0; j < n_features; ++j) {
        grad_w[j] = weights_[j];
    }
    
    // Loop through training examples
    for (size_t i = 0; i < n_samples; ++i) {
        double f = bias_;
        for (size_t j = 0; j < n_features; ++j) {
            f += weights_[j] * X[i][j];
        }
        double margin = y[i] * f;
        
        if (margin < 1.0) {
            loss += C * (1.0 - margin);
            
            for (size_t j = 0; j < n_features; ++j) {
                grad_w[j] -= C * y[i] * X[i][j];
            }

            grad_b -= C * y[i];
        }
    }
    
    return loss;
}

#pragma endregion

} // namespace gdv::models