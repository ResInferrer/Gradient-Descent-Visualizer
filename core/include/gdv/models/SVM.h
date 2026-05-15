#pragma once

#include "gdv/losses/HingeLoss.h"  
#include <vector>
#include <memory>

namespace gdv::models{

class SVM {
public:
    SVM();
    
    // Methods of models
    void fit(const std::vector<std::vector<double>>& X,
            const std::vector<int>& y,
            double C = 1.0,
            double learning_rate = 0.01,
            int iterations = 1000,
            bool verbose = false);

    double decision_function(const std::vector<double>& X_row) const;
    int predict(const std::vector<double>& X_row) const;  
    
private:
    double hinge_loss_and_subgradient(const std::vector<std::vector<double>>& X,
                                    const std::vector<int>& y,
                                    double C,
                                    std::vector<double>& grad_w,
                                    double& grad_b) const;
    
    std::vector<double> weights_;
    double bias_;
    std::unique_ptr<gdv::losses::ILossFunction> loss_fn_;
};

} // namespace gdv::models