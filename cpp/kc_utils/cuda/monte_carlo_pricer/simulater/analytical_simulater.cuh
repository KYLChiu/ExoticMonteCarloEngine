#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <memory>
#include <type_traits>
#include "../model/black_scholes.cuh"
#include "simulater.cuh"

namespace kcu::mc {

class analytical_simulater : public simulater<analytical_simulater> {
   public:
    __host__ __device__ analytical_simulater(double T) : T_(T) {}

    template <typename Model>
    __host__ double simulate_cpp_impl(std::shared_ptr<Model> model,
                                      std::size_t seed) const {
        using model_t = std::decay_t<Model>;
        static_assert(
            std::is_same_v<model_t, black_scholes>,
            "The analytical solver does not support the input solver.");
        if constexpr (std::is_same_v<model_t, black_scholes>) {
            return bs_impl(model, seed);
        }
        // Deliberately leave this code path undefined.
    }

    template <typename Model>
    __device__ double simulate_cuda_impl(thrust::device_ptr<Model> model,
                                         std::size_t seed) const {
        using model_t = std::decay_t<Model>;
        static_assert(
            std::is_same_v<model_t, black_scholes>,
            "The analytical solver does not support the input solver.");
        if constexpr (std::is_same_v<model_t, black_scholes>) {
            return bs_impl(model, seed);
        }
        // Deliberately leave this code path undefined.
    }

   private:
    template <typename ModelPtr>
    __host__ __device__ double bs_impl(ModelPtr model, std::size_t seed) const {
        thrust::default_random_engine rng(seed);
        thrust::random::normal_distribution<double> dist(0.0, sqrtf(T_));
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20014
        double S_0 = model->initial_value();
        double r = model->r();
        double sigma = model->sigma();
        double W_T = dist(rng);
        return S_0 * exp((r - sigma * sigma / 2.0) * T_ + sigma * W_T);
#pragma nv_diagnostic pop
    }

    double T_;
};
}  // namespace kcu::mc
