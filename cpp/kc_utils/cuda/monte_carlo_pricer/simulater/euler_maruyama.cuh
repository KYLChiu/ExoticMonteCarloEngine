#pragma once

#include <cuda_runtime.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include "simulater.cuh"

namespace kcu::mc {

class euler_maruyama : public simulater<euler_maruyama> {
   public:
    __host__ __device__ euler_maruyama(std::size_t num_steps, double T)
        : num_steps_(num_steps), T_(T) {}

    template <typename Model>
    __host__ double simulate_cpp_impl(std::shared_ptr<Model> model,
                                      std::size_t rng_seed) const {
        return simulate_impl(model, rng_seed);
    }

    template <typename Model>
    __device__ double simulate_cuda_impl(thrust::device_ptr<Model> model,
                                         std::size_t rng_seed) const {
        return simulate_impl(model, rng_seed);
    }

   private:
    template <typename ModelPtr>
    __host__ __device__ double simulate_impl(ModelPtr model,
                                             std::size_t rng_seed) const {
        double dt = T_ / num_steps_;
        double std_dev = sqrtf(dt);

        thrust::default_random_engine rng(rng_seed);
        thrust::random::normal_distribution<double> dist(0.0, std_dev);

#pragma nv_diagnostic push
#pragma nv_diag_suppress 20014
        double X_t = model->initial_value();
        double t = 0.0;
        for (std::size_t i = 0; i < num_steps_; ++i) {
            double dW_t = dist(rng);
            double drift = model->drift(X_t, t);
            double diffusion = model->diffusion(X_t, t);
            X_t = X_t + drift * dt + diffusion * dW_t;
            t += dt;
        }
#pragma nv_diagnostic pop
        return X_t;
    }

    std::size_t num_steps_;
    double T_;
};
}  // namespace kcu::mc
