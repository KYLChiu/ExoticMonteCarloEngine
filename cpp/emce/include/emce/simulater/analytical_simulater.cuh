#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <memory>
#include <type_traits>
#include "emce/model/black_scholes.cuh"
#include "emce/option/path_dependent_option.cuh"
#include "emce/simulater/simulater.cuh"

namespace emce {

class analytical_simulater : public simulater<analytical_simulater> {
    friend class simulater<analytical_simulater>;

   public:
    __host__ __device__ analytical_simulater() {}

   private:
    template <typename Option, typename Model>
    __host__ double simulate_cpp_impl(std::shared_ptr<Option> option,
                                      std::shared_ptr<Model> model,
                                      std::size_t seed) const {
        if constexpr (std::is_same_v<Model, black_scholes>) {
            return bs_impl(option, model, seed);
        }
        // Deliberately leave this code path undefined.
    }

    template <typename Option, typename Model>
    __device__ double simulate_cuda_impl(thrust::device_ptr<Option> option,
                                         thrust::device_ptr<Model> model,
                                         std::size_t seed) const {
        if constexpr (std::is_same_v<Model, black_scholes>) {
            return bs_impl(option, model, seed);
        }
        // Deliberately leave this code path undefined.
    }

    template <typename OptionPtr, typename ModelPtr>
    __host__ __device__ double bs_impl(OptionPtr option, ModelPtr model,
                                       std::size_t seed) const {
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20014
        using option_t = std::decay_t<decltype(*option.get())>;
        static_assert(
            ! std::is_base_of_v<path_dependent_option<option_t>, option_t>,
            "Analytical simulater does not support path dependent "
            "options.");

        double T = model->parameter(black_scholes::parameters::maturity);

        thrust::default_random_engine rng(seed);
        thrust::random::normal_distribution<double> dist(0.0, sqrtf(T));

        double S_0 = model->parameter(black_scholes::parameters::initial_value);
        double r = model->parameter(black_scholes::parameters::risk_free_rate);
        double sigma = model->parameter(black_scholes::parameters::volatility);
        double W_T = dist(rng);
        double drift = (r - sigma * sigma / 2.0) * T;
        double diffusion = sigma * W_T;
        double common = S_0 * exp(drift);
        // Antithetic variates:
        // https://en.wikipedia.org/wiki/Antithetic_variates
        double S_T = common * exp(diffusion);
        double S_Ta = common * exp(-diffusion);
        return (option->payoff(S_T) + option->payoff(S_Ta)) / 2.0;
#pragma nv_diagnostic pop
    }
};
}  // namespace emce
