#pragma once

#include "gdv/losses/BinaryCrossEntropy.h"
#include <vector>
#include <memory>

namespace gdv::models{

struct GradientResult {
    double loss;
    std::vector<double> grad_weights;
    double grad_bias;
};

class LogisticRegression {
public:
    LogisticRegression();
    
    // Methods of models
    void fit(const std::vector<std::vector<double>>& X,
            const std::vector<double>& y,
            double learning_rate = 0.01,
            int iterations = 1000,
            double l2_lambda = 0.0,
            bool verbose = false);
    int predict(const std::vector<double>& X_row, double threshold = 0.5) const;
    std::vector<double> predict_proba(const std::vector<double>& X_row) const;

private:
    double sigmoid(double z) const;
    GradientResult compute_loss_and_gradient(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y_true) const; 

    std::vector<double> weights_;
    double bias_;  
    std::unique_ptr<gdv::losses::ILossFunction> loss_fn_;
};

} // namespace gdv::models