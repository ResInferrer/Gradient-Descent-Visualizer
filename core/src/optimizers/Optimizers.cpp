#include "gdv/optimizers/Optimizers.h"
#include <cmath>

namespace gdv::optimizers {

Optimizers::Optimizers() 
{
    // nothing else to initialize for now
}

double Optimizers::L1_Norm(const std::vector<double>& omegas) {
    double sum = 0.0;

    for (double val : omegas) {
        sum += std::abs(val);
    }

    return sum;
}

double Optimizers::L2_Norm(const std::vector<double>& omegas) {
    double sum_sq = 0.0;

    for (double val : omegas) {
        sum_sq += val * val;
    }

    return std::sqrt(sum_sq);
}

}  // namespace gdv::optimizers
