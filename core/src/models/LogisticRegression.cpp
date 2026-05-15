#include "gdv/models/LogisticRegression.h"
#include <cmath>
#include <cassert>

namespace gdv::models{

LogisticRegression::LogisticRegression() 
    : weights_(), bias_(0.0) 
{
}

#pragma region Public API

void LogisticRegression::fit(const std::vector<std::vector<double>>& X,
                             const std::vector<double>& y,
                             double learning_rate,
                             int iterations,
                             double l2_lambda,
                             bool verbose) {
    assert(X.size() == y.size() && "Feature count mismatch");

    size_t n_samples = X.size();
    if (n_samples == 0) return;
    size_t n_features = X[0].size();
    assert(y.size() == n_samples);

    weights_.assign(n_features, 0.0);
    bias_ = 0.0;
    
    for (int iter = 0; iter < iterations; ++iter) {
        // Calculate gradients and current loss
        GradientResult grad = compute_loss_and_gradient(X, y);
        
        // Add L2 regularization to weight gradients (but not to bias)
        for (size_t j = 0; j < n_features; ++j) {
            grad.grad_weights[j] += l2_lambda * weights_[j];
        }
        
        // Update parameters (gradient descent)
        for (size_t j = 0; j < n_features; ++j) {
            weights_[j] -= learning_rate * grad.grad_weights[j];
        }
        bias_ -= learning_rate * grad.grad_bias;
    }
}

std::vector<double> LogisticRegression::predict_proba(const std::vector<double>& X_row) const {
    assert(weights_.size() == X_row.size() && "Feature count mismatch");

    double linear = bias_; 
    for (size_t i = 0; i < weights_.size(); ++i) {
        linear += weights_[i] * X_row[i];
    }

    return {1.0 - sigmoid(linear), sigmoid(linear)};
}

int LogisticRegression::predict(const std::vector<double>& X_row, double threshold) const {
    assert(weights_.size() == X_row.size() && "Feature count mismatch");

    double z = bias_;
    for (size_t i = 0; i < X_row.size(); ++i) {
        z += weights_[i] * X_row[i];
    }

    double p = sigmoid(z);
    return (p >= threshold) ? 1 : 0;
}

#pragma endregion

#pragma region Private API

GradientResult LogisticRegression::compute_loss_and_gradient(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y_true) const 
{
    size_t n_samples = X.size();
    size_t n_features = (n_samples == 0) ? 0 : X[0].size();
    
    GradientResult result;
    result.loss = 0.0;
    result.grad_weights.assign(n_features, 0.0);
    result.grad_bias = 0.0;
    
    for (size_t i = 0; i < n_samples; ++i) {
        // Calculate linear combination
        double z = bias_;
        for (size_t j = 0; j < n_features; ++j) {
            z += weights_[j] * X[i][j];
        }
        double prob = sigmoid(z);
        
        // Cross-entropy loss
        result.loss += - (y_true[i] * std::log(prob + 1e-15) + 
                         (1 - y_true[i]) * std::log(1 - prob + 1e-15));
        
        // Gradients (without regularization)
        double error = prob - y_true[i];
        for (size_t j = 0; j < n_features; ++j) {
            result.grad_weights[j] += error * X[i][j];
        }
        result.grad_bias += error;
    }
    
    // Averaging
    double inv_n = 1.0 / n_samples;
    result.loss *= inv_n;
    for (size_t j = 0; j < n_features; ++j) {
        result.grad_weights[j] *= inv_n;
    }
    result.grad_bias *= inv_n;
    
    return result;
}

double LogisticRegression::sigmoid(double z) const {
    // Overflow protection exp(-z) for large negative z
    if (z >= 0) {
        return 1.0 / (1.0 + std::exp(-z));
    } else {
        double exp_z = std::exp(z);
        return exp_z / (1.0 + exp_z);
    }
}

#pragma endregion

}  // namespace gdv::models
