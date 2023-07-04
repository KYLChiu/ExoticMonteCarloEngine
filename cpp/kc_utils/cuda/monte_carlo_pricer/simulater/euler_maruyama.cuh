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
    double simulate_cpp_impl(const Model& model, std::size_t rng_seed) const {
        double dt = T_ / num_steps_;
        double std_dev = std::sqrt(dt);
        std::mt19937 rng{rng_seed};
        std::normal_distribution<double> dist(0, std_dev);
        double X_t = model.initial_value();
        double t = 0.0;
        for (std::size_t i = 0; i < num_steps_; ++i) {
            double dW_t = dist(rng);
            X_t =
                X_t + model.drift(X_t, t) * dt + model.diffusion(X_t, t) * dW_t;
            t += dt;
        }
        return X_t;
    }

    template <typename ModelPtr>
    __device__ double simulate_cuda_impl(ModelPtr model,
                                         std::size_t rng_seed) const {
        double dt = T_ / num_steps_;
        double std_dev = sqrtf(dt);

        thrust::default_random_engine rng(rng_seed);
        thrust::random::normal_distribution<double> dist(0.0, std_dev);
        double X_t = model->initial_value();
        double t = 0.0;
        for (std::size_t i = 0; i < num_steps_; ++i) {
            double dW_t = dist(rng);
            X_t = X_t + model->drift(X_t, t) * dt +
                  model->diffusion(X_t, t) * dW_t;
            t += dt;
        }
        return X_t;
    }

   private:
    std::size_t num_steps_;
    double T_;
};
}  // namespace kcu::mc
