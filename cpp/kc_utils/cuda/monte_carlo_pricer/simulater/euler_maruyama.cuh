#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <memory>
#include <type_traits>
#include "../option/path_dependent_option.cuh"
#include "simulater.cuh"

namespace kcu::mc {

namespace detail {

    __host__ __device__ std::size_t greatest_common_divisor(std::size_t a,
                                                            std::size_t b) {
        while (b != 0) {
            std::size_t temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    __host__ __device__ std::size_t lowest_common_multiple(std::size_t a,
                                                           std::size_t b) {
        std::size_t gcd = greatest_common_divisor(a, b);
        return (a / gcd) * b;
    }

}  // namespace detail

class euler_maruyama : public simulater<euler_maruyama> {
    friend class simulater<euler_maruyama>;

   public:
    __host__ __device__ euler_maruyama(std::size_t num_steps, double T)
        : num_steps_(num_steps), T_(T) {}

   private:
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

    template <typename OptionPtr, typename ModelPtr>
    __host__ __device__ double simulate_impl(OptionPtr option, ModelPtr model,
                                             std::size_t rng_seed) const {
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20014

        using option_t = std::decay_t<decltype(*option.get())>;
        thrust::default_random_engine rng(rng_seed);

        // Antithetic variates
        double X_t = model->initial_value();
        double X_ta = model->initial_value();
        double t = 0.0;

        if constexpr (std::is_base_of_v<path_dependent_option<option_t>,
                                        option_t>) {
            std::size_t periods = option->periods();
            std::size_t num_steps =
                detail::lowest_common_multiple(num_steps_, periods);
            std::size_t period = num_steps / periods;
            double dt = T_ / num_steps;
            double std_dev = sqrtf(dt);

            double* X_ts = new double[periods];
            double* X_tas = new double[periods];

            thrust::random::normal_distribution<double> dist(0.0, std_dev);
            std::size_t idx = 0;

            for (std::size_t i = 1; i <= num_steps; ++i) {
                double dW_t = dist(rng);
                X_t = X_t + model->drift(X_t, t) * dt +
                      model->diffusion(X_t, t) * dW_t;
                X_ta = X_ta + model->drift(X_ta, t) * dt -
                       model->diffusion(X_ta, t) * dW_t;
                t += dt;
                if (i % period == 0) {
                    X_ts[idx] = X_t;
                    X_tas[idx++] = X_ta;
                }
            }

            auto path = (option->payoff(X_ts) + option->payoff(X_tas)) / 2.0;

            delete[] X_ts;
            delete[] X_tas;

            return path;
        } else {
            double dt = T_ / num_steps_;
            double std_dev = sqrtf(dt);

            thrust::random::normal_distribution<double> dist(0.0, std_dev);
            for (std::size_t i = 0; i < num_steps_; ++i) {
                double dW_t = dist(rng);
                X_t = X_t + model->drift(X_t, t) * dt +
                      model->diffusion(X_t, t) * dW_t;
                X_ta = X_ta + model->drift(X_ta, t) * dt -
                       model->diffusion(X_ta, t) * dW_t;
                t += dt;
            }
            return (option->payoff(X_t) + option->payoff(X_ta)) / 2.0;
        }
    }
#pragma nv_diagnostic pop

    std::size_t num_steps_;
    double T_;
};
}  // namespace kcu::mc
