#pragma once

#include <complex>
#include <cstddef>

namespace taco::bath {

class CorrelationFunction {
public:
    virtual ~CorrelationFunction() = default;
    virtual std::complex<double> operator()(double tau, std::size_t alpha, std::size_t beta) const = 0;
    virtual std::size_t rank() const noexcept = 0;
};

}  // namespace taco::bath

