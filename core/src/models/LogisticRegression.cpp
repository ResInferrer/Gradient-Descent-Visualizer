#include "gdv/models/LogisticRegression.h"
#include <cmath>
#include <cassert>

namespace gdv::models{

LogisticRegression::LogisticRegression() 
    : weights_(), bias_(0.0),
      loss_fn_(std::make_unique<gdv::losses::BinaryCrossEntropy>()) {}

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
    size_t n_features = X.empty() ? 0 : X[0].size();
    
    std::vector<double> proba(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        double z = bias_;
        for (size_t j = 0; j < n_features; ++j)
            z += weights_[j] * X[i][j];
        proba[i] = sigmoid(z);
    }
    
    double loss = loss_fn_->compute(y_true, proba);
    std::vector<double> grad_proba = loss_fn_->gradient(y_true, proba);
    
    std::vector<double> grad_weights(n_features, 0.0);
    double grad_bias = 0.0;
    for (size_t i = 0; i < n_samples; ++i) {
        double p = proba[i];
        double dz = grad_proba[i] * p * (1.0 - p);
        grad_bias += dz;
        for (size_t j = 0; j < n_features; ++j)
            grad_weights[j] += dz * X[i][j];
    }
    
    GradientResult result;
    result.loss = loss;
    result.grad_weights = std::move(grad_weights);
    result.grad_bias = grad_bias;
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
