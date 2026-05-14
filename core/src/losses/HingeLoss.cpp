#include "gdv/losses/HingeLoss.h"
#include <cmath>
#include <cassert>

namespace gdv::losses {

double HingeLoss::compute(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred) const {
            
        assert(y_true.size() == y_pred.size());
        size_t n = y_true.size();
        double sum = 0.0;

        for (size_t i = 0; i < n; ++i) {
                double margin = 1.0 - y_true[i] * y_pred[i];
                if (margin > 0.0) {
                sum += margin;
                }
        }

        return sum / n;
}

std::vector<double> HingeLoss::gradient(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred) const {

        assert(y_true.size() == y_pred.size());
        size_t n = y_true.size();
        std::vector<double> grad(n);
        double factor = -1.0 / n;

        for (size_t i = 0; i < n; ++i) {
                if (y_true[i] * y_pred[i] < 1.0) {
                grad[i] = factor * y_true[i];
                } else {
                grad[i] = 0.0;
                }
        }

        return grad;
}

}  // namespace gdv::losses
