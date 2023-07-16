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

// EM method for simulating a generic first order SDE:
// https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
class euler_maruyama : public simulater<euler_maruyama> {
    friend class simulater<euler_maruyama>;

   public:
    __host__ __device__ euler_maruyama(std::size_t num_steps)
        : num_steps_(num_steps) {}

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

    __host__ __device__ std::size_t greatest_common_divisor(
        std::size_t a, std::size_t b) const {
        while (b != 0) {
            std::size_t temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    __host__ __device__ std::size_t lowest_common_multiple(
        std::size_t a, std::size_t b) const {
        std::size_t gcd = greatest_common_divisor(a, b);
        return (a / gcd) * b;
    }

    template <typename OptionPtr, typename ModelPtr>
    __host__ __device__ double bs_impl(OptionPtr option, ModelPtr model,
                                       std::size_t rng_seed) const {
#pragma nv_diagnostic push
#pragma nv_diag_suppress 20014

        using option_t = std::decay_t<decltype(*option.get())>;
        thrust::default_random_engine rng(rng_seed);

        // Antithetic variates:
        // https://en.wikipedia.org/wiki/Antithetic_variates
        double X_t = model->parameter(black_scholes::parameters::initial_value);
        double X_ta = X_t;

        double T = model->parameter(black_scholes::parameters::maturity);
        double t = 0.0;

        if constexpr (std::is_base_of_v<path_dependent_option<option_t>,
                                        option_t>) {
            std::size_t periods = option->periods();

            // Food for thought: the current design has a generic periods
            // fields for path dependent options (this means
            // monitoring periods for Asian option, and how many times we sample
            // the path to check against the barrier for Barriers). But if the
            // number of periods is too low, we cannot use that as an accurate
            // step in simulating SDE. So users additionally pass in the normal
            // num_steps_ field and we take steps = LCM(num_steps_,  periods).
            // Is this overcomplicated?
            std::size_t num_steps = lowest_common_multiple(num_steps_, periods);
            std::size_t period = num_steps / periods;
            double dt = T / num_steps;

            // TODO: To be both CUDA and C++ compliant we must use raw ptrs
            // here, but need to consider exception safety in future.
            double* X_ts = new double[periods];
            double* X_tas = new double[periods];

            thrust::random::normal_distribution<double> dist(0.0, sqrtf(dt));
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
            double dt = T / num_steps_;
            thrust::random::normal_distribution<double> dist(0.0, sqrtf(dt));

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
};
}  // namespace emce
