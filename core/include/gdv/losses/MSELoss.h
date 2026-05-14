#pragma once

#include "ILossFunction.h"

namespace gdv::losses {

class MSELoss : public ILossFunction {
public:
   double compute(const std::vector<double>& y_true,
                  const std::vector<double>& y_pred) const override;

    std::vector<double> gradient(const std::vector<double>& y_true,
                                 const std::vector<double>& y_pred) const override;
};

}  // namespace gdv::losses