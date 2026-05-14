#pragma once

#include <vector>

namespace gdv::losses {

class ILossFunction {
public:
    virtual ~ILossFunction() = default; 

    // Calculate the value of the loss function
    virtual double compute(const std::vector<double>& y_true,
                           const std::vector<double>& y_pred) const = 0;

    // Calculate gradient from predictions
    virtual std::vector<double> gradient(const std::vector<double>& y_true,
                                         const std::vector<double>& y_pred) const = 0;
};

}  // namespace gdv::losses