#pragma once

/* A Monte Carlo Pricer implemented using:
 * - CUDA (via Thrust)
 * - C++ (multi-threaded).
 */

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <execution>
#include <functional>
#include <future>
#include <memory>
#include <random>
#include <thread>
#include "dispatch_type.hpp"
#include "model/model.cuh"
#include "option/option.cuh"
#include "simulater/simulater.cuh"

namespace kcu::mc {

template <dispatch_type DispatchType>
class monte_carlo_pricer final {
   private:
   public:
    monte_carlo_pricer(std::size_t num_paths) : num_paths_(num_paths) {}

    template <typename Option, typename Simulater, typename Model>
    double run(std::shared_ptr<Option> opt, std::shared_ptr<Simulater> sim,
               std::shared_ptr<Model> mdl, double T) {
        if constexpr (DispatchType == dispatch_type::cpp) {
            auto num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> ts;
            ts.reserve(num_threads);

            // Store the number of simulations per thread, the remainder goes to
            // the last thread.
            std::vector<std::size_t> num_sims(num_threads,
                                              num_paths_ / num_threads);
            num_sims[num_sims.size() - 1] += num_paths_ % num_threads;

            std::vector<aligned_double> thread_res(num_threads, 0.0);

            auto work = [&](std::size_t thread_id) {
                auto sims = num_sims[thread_id];
                auto seed = sims * thread_id;
                for (std::size_t j = 0; j < sims; ++j) {
                    auto S = sim->simulate_cpp(mdl, seed++);
                    thread_res[thread_id].value_ += opt->payoff(S);
                }
            };

            for (std::size_t i = 0; i < num_threads; ++i) {
                auto t = std::thread(work, i);
                ts.emplace_back(std::move(t));
            }
            for (auto& t : ts) {
                t.join();
            }

            double sum = 0.0;
            for (const auto& res : thread_res) {
                sum += res.value_;
            }

            return mdl->discount_factor(T) * sum / num_paths_;

        } else {
            // Copy to device data
            auto opt_dv = thrust::device_vector<Option>(1, *opt);
            auto sim_dv = thrust::device_vector<Simulater>(1, *sim);
            auto mdl_dv = thrust::device_vector<Model>(1, *mdl);
            auto opt_d = opt_dv.data();
            auto sim_d = sim_dv.data();
            auto mdl_d = mdl_dv.data();

            thrust::device_vector<double> input(num_paths_);
            thrust::sequence(input.begin(), input.end(), 0, 1);

            auto sum = thrust::transform_reduce(
                thrust::device, input.begin(), input.end(),
                [opt_d, sim_d, mdl_d] __device__(double idx) {
                    auto S = sim_d->simulate_cuda(mdl_d, idx);
                    return opt_d->payoff(S);
                },
                0.0, thrust::plus<double>());

            return mdl->discount_factor(T) * sum / num_paths_;
        }
    }

   private:
    // Wrapper class to alleviate effects of false sharing
    struct aligned_double {
        aligned_double(double value) : value_(value) {}
        alignas(64) double value_;
    };

    std::size_t num_paths_;
};

}  // namespace kcu::mc
