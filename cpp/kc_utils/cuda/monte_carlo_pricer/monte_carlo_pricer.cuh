#pragma once

/* A Monte Carlo Pricer implemented using:
 * - CUDA (via Thrust)
 * - C++ (multi-threaded).
 */

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <memory>
#include <thread>
#include <type_traits>
#include "dispatch_type.hpp"
#include "model/model.cuh"
#include "option/option.cuh"
#include "option/path_dependent_option.cuh"
#include "simulater/simulater.cuh"

namespace kcu::mc {

namespace detail {

    // Functor for running the simulater.
    // NB: this must be public because of visibility restrictions with
    // thrust::reduce.
    template <typename Option, typename Simulater, typename Model>
    struct simulate_cuda {
        __host__ simulate_cuda(thrust::device_ptr<Option> opt_d,
                               thrust::device_ptr<Simulater> sim_d,
                               thrust::device_ptr<Model> mdl_d)
            : opt_d_(opt_d), sim_d_(sim_d), mdl_d_(mdl_d) {}

        __device__ double operator()(double idx) const {
            return sim_d_->simulate_cuda(opt_d_, mdl_d_, idx);
        }

        thrust::device_ptr<Option> opt_d_;
        thrust::device_ptr<Simulater> sim_d_;
        thrust::device_ptr<Model> mdl_d_;
    };

}  // namespace detail

template <dispatch_type DispatchType>
class monte_carlo_pricer final {
   public:
    __host__ monte_carlo_pricer(std::size_t num_paths)
        : num_paths_(num_paths) {}

    template <typename Option, typename Simulater, typename Model>
    __host__ double run(std::shared_ptr<Option> opt,
                        std::shared_ptr<Simulater> sim,
                        std::shared_ptr<Model> mdl, double T) {
        static_assert(std::is_base_of_v<option<Option>, Option>,
                      "Unexpected option type. Options are expected to derived "
                      "from the CRTP type option<Derived>.");
        static_assert(std::is_base_of_v<simulater<Simulater>, Simulater>,
                      "Unexpected simulater type. Simulaters are expected to "
                      "derived from the CRTP type simulater<Derived>.");
        static_assert(std::is_base_of_v<model<Model>, Model>,
                      "Unexpected model type. Models are expected to derived "
                      "from the CRTP type model<Derived>.");
        if constexpr (DispatchType == dispatch_type::cpp) {
            return dispatch_cpp(opt, sim, mdl, T);
        } else {
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
            return dispatch_cuda(opt, sim, mdl, T);
        }
    }

   private:
    // Wrapper for double to alleviate effects of false sharing
    struct aligned_double {
        aligned_double(double value) : value_(value) {}
        alignas(64) double value_;
    };

    template <typename Option, typename Simulater, typename Model>
    double dispatch_cpp(std::shared_ptr<Option> opt,
                        std::shared_ptr<Simulater> sim,
                        std::shared_ptr<Model> mdl, double T) {
        std::size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> ts;
        ts.reserve(num_threads);

        std::vector<aligned_double> thread_res(num_threads, 0.0);

        auto work = [&](std::size_t thread_id) {
            std::size_t sims = num_paths_ / num_threads;
            std::size_t seed = sims * thread_id;
            // The last thread gets the remainder of the paths.
            if (thread_id == num_threads - 1) {
                sims += num_paths_ % num_threads;
            }
            for (std::size_t j = 0; j < sims; ++j) {
                thread_res[thread_id].value_ +=
                    sim->simulate_cpp(opt, mdl, seed++);
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
    }

    template <typename Option, typename Simulater, typename Model>
    double dispatch_cuda(std::shared_ptr<Option> opt,
                         std::shared_ptr<Simulater> sim,
                         std::shared_ptr<Model> mdl, double T) {
        // Copy data onto device
        auto opt_dv = thrust::device_vector<Option>(1, *opt);
        auto sim_dv = thrust::device_vector<Simulater>(1, *sim);
        auto mdl_dv = thrust::device_vector<Model>(1, *mdl);
        auto opt_d = opt_dv.data();
        auto sim_d = sim_dv.data();
        auto mdl_d = mdl_dv.data();

        thrust::device_vector<double> input(num_paths_);
        thrust::sequence(input.begin(), input.end(), 0, 1);

        auto sum =
            thrust::transform_reduce(thrust::device, input.begin(), input.end(),
                                     detail::simulate_cuda(opt_d, sim_d, mdl_d),
                                     0.0, thrust::plus<double>());

        return mdl->discount_factor(T) * sum / num_paths_;
    }

    std::size_t num_paths_;
};

}  // namespace kcu::mc
