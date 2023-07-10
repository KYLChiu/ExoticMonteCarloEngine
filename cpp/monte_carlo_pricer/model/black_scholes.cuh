#pragma once

#include <cuda_runtime.h>
#include "model.cuh"

namespace emce {

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

    __host__ static constexpr std::size_t num_parameters() { return 4; }

   private:
    __host__ __device__ double drift_impl(double X_t, double t) const {
        return base_t::parameters_[1] * X_t;
    }
    __host__ __device__ double diffusion_impl(double X_t, double t) const {
        return base_t::parameters_[2] * X_t;
    }
    __host__ __device__ double discount_factor_impl() const {
        return exp(-base_t::parameters_[1] * base_t::parameters_[3]);
    }
    __host__ __device__ double parameter_impl(parameters parameter) const {
        return base_t::parameters_[static_cast<int>(parameter)];
    }
    __host__ std::pair<std::shared_ptr<black_scholes>, double> bump_impl(
        sensitivities sensitivity, double by_factor) const {
        auto mdl = std::make_shared<black_scholes>(*this);
        std::size_t idx = static_cast<int>(sensitivity);
        double param = mdl->parameter(parameters(idx));
        double bump_size = param * by_factor;
        double new_param = param + bump_size;
        mdl->base_t::parameters_[idx] = new_param;
        return {mdl, bump_size};
    }
};

}  // namespace emce