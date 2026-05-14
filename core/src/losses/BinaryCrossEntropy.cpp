#include "gdv/losses/BinaryCrossEntropy.h"
#include <cmath>
#include <cassert>

namespace gdv::losses {

double BinaryCrossEntropy::compute(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) const {
    
    assert(y_true.size() == y_pred.size());
    size_t n = y_true.size();
    const double eps = 1e-15;
    double loss = 0.0;

    for (size_t i = 0; i < n; ++i) {
            double pred = y_pred[i];
            double true_val = y_true[i];
            // Clipping to prevent log(0)
            double pred_clipped = std::max(eps, std::min(1.0 - eps, pred));
            loss += - (true_val * std::log(pred_clipped) + (1.0 - true_val) * std::log(1.0 - pred_clipped));
    }

    return loss / static_cast<double>(n);
}

std::vector<double> BinaryCrossEntropy::gradient(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) const {

    assert(y_true.size() == y_pred.size());
    size_t n = y_true.size();
    const double eps = 1e-15;
    std::vector<double> grad(n);

    for (size_t i = 0; i < n; ++i) {
            double pred = y_pred[i];
            double true_val = y_true[i];
            // Clipping for denominator
            double pred_clipped = std::max(eps, std::min(1.0 - eps, pred));
            double denom = pred_clipped * (1.0 - pred_clipped);
            // dL/dpred = (pred - true) / (n * denom)
            grad[i] = (pred_clipped - true_val) / (static_cast<double>(n) * denom);
    }

    return grad;
}

}  // namespace gdv::losses
