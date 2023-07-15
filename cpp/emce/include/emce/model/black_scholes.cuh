#pragma once

#include <memory>
#include "emce/model/model.cuh"

namespace emce {

// Black Scholes model:
// https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
// This currently does not yet support dividend or repo rates.
class black_scholes final : public model<black_scholes, 4> {
    using base_t = model<black_scholes, 4>;
    friend class model<black_scholes, 4>;

   public:
    enum class parameters : int {
        initial_value,
        risk_free_rate,
        volatility,
        maturity
    };
    enum class sensitivities : int { delta, rho, vega, theta };

    // dS_t = rS_t dt + sigma S_t dW_t
    // https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    __host__ __device__ explicit black_scholes(double S_0 /*initial value*/,
                                               double r /*risk-free rate*/,
                                               double sigma /*volatility*/,
                                               double T /* maturity */)
        : base_t(S_0, r, sigma, T) {}

    __host__ __device__ static constexpr std::size_t num_parameters() {
        return 4;
    }

   private:
    __host__ __device__ double drift_impl(double X_t, double t) const;
    __host__ __device__ double diffusion_impl(double X_t, double t) const;
    __host__ __device__ double discount_factor_impl() const;
    __host__ __device__ double parameter_impl(parameters parameter) const;
    __host__ std::shared_ptr<black_scholes> bump_impl(sensitivities sensitivity,
                                                      double bump_size) const;
};

}  // namespace emce