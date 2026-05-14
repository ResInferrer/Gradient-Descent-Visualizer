#include "gdv/losses/MSELoss.h"
#include <cmath>
#include <cassert>

namespace gdv::losses {

double MSELoss::compute(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) const {

    assert(y_true.size() == y_pred.size());
    size_t n = y_true.size();
    double counter_sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        double diff = y_pred[i] - y_true[i];
        counter_sum += diff * diff;
    }

    return counter_sum / n;
}

std::vector<double> MSELoss::gradient(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) const {

    assert(y_true.size() == y_pred.size());
    size_t n = y_true.size();
    std::vector<double> grad(n);
    double factor = 2.0 / n;

    for (size_t i = 0; i < n; ++i) {
        grad[i] = factor * (y_pred[i] - y_true[i]);
    }

    return grad;        
}

}  // namespace gdv::losses
