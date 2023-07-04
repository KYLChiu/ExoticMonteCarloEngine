#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <memory>
#include "simulater.cuh"

namespace kcu::mc {

class euler_maruyama : public simulater<euler_maruyama> {
   public:
    __host__ __device__ euler_maruyama(std::size_t num_steps, double T)
        : num_steps_(num_steps), dt_(T / num_steps_), std_dev_(sqrtf(dt_)) {}

    template <typename Option, typename Model>
    __host__ double simulate_cpp_impl(std::shared_ptr<Option> option,
                                      std::shared_ptr<Model> model,
                                      std::size_t rng_seed) const {
        return simulate_impl(option, model, rng_seed);
    }

    template <typename Option, typename Model>
    __device__ double simulate_cuda_impl(thrust::device_ptr<Option> option,
                                         thrust::device_ptr<Model> model,
                                         std::size_t rng_seed) const {
        return simulate_impl(option, model, rng_seed);
    }

   private:
    template <typename OptionPtr, typename ModelPtr>
    __host__ __device__ double simulate_impl(OptionPtr option, ModelPtr model,
                                             std::size_t rng_seed) const {
        thrust::default_random_engine rng(rng_seed);
        thrust::random::normal_distribution<double> dist(0.0, std_dev_);

#pragma nv_diagnostic push
#pragma nv_diag_suppress 20014
        // Antithetic variate pair
        double X_t = model->initial_value();
        double X_ta = model->initial_value();
        double t = 0.0;
        for (std::size_t i = 0; i < num_steps_; ++i) {
            double dW_t = dist(rng);
            X_t = X_t + model->drift(X_t, t) * dt_ +
                  model->diffusion(X_t, t) * dW_t;
            X_ta = X_ta + model->drift(X_ta, t) * dt_ -
                   model->diffusion(X_ta, t) * dW_t;
            t += dt_;
        }
        return (option->payoff(X_t) + option->payoff(X_ta)) / 2.0;
    }
#pragma nv_diagnostic pop

    std::size_t num_steps_;
    double dt_;
    double std_dev_;
};
}  // namespace kcu::mc
