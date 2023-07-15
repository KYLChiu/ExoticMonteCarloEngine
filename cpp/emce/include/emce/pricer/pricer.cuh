#pragma once

/* A Monte Carlo Pricer implemented using:
 * - CUDA (via Thrust)
 * - C++ (multi-threaded).
 */

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <memory>
#include <thread>
#include <type_traits>
#include "emce/model/model.cuh"
#include "emce/option/option.cuh"
#include "emce/option/path_dependent_option.cuh"
#include "emce/pricer/dispatch_type.cuh"
#include "emce/simulater/simulater.cuh"

namespace emce {

namespace detail {

    // Functor for running the simulater.
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
    __host__ double price(std::shared_ptr<Option> opt,
                          std::shared_ptr<Simulater> sim,
                          std::shared_ptr<Model> mdl) {
        static_assert(std::is_base_of_v<option<Option>, Option>,
                      "Unexpected option type. Options are expected to derived "
                      "from the CRTP type option<Derived>.");
        static_assert(std::is_base_of_v<simulater<Simulater>, Simulater>,
                      "Unexpected simulater type. Simulaters are expected to "
                      "derived from the CRTP type simulater<Derived>.");
        static_assert(
            std::is_base_of_v<model<Model, Model::num_parameters()>, Model>,
            "Unexpected model type. Models are expected to derived "
            "from the CRTP type model<Derived, num_parameters>.");
        if constexpr (DispatchType == dispatch_type::cpp) {
            return price_cpp(opt, sim, mdl);
        } else {
            // Set the CUDA heap size to 256MB - we need this for carrying the
            // paths around in device memory (for path dependent options).
            cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                               static_cast<std::size_t>(2.56e+8));
            return price_cuda(opt, sim, mdl);
        }
    }

    template <typename Option, typename Simulater, typename Model,
              typename Sensitivity>
    __host__ double sensitivity(std::shared_ptr<Option> opt,
                                std::shared_ptr<Simulater> sim,
                                std::shared_ptr<Model> mdl,
                                Sensitivity sensitivity,
                                double bump_size = 1e-6) {
        // Central finite differencing with absolute bump size.
        // https://people.maths.ox.ac.uk/gilesm/mc/module_2/module_2_2.pdf
        // Long-term TODO: AAD. But this isn't so easy with CUDA!
        // TODO: second-order sensitivities.
        // TODO: handle discontinuous first derivatives.
        // TODO: multiplicative bumps.
        // Note: below prices actually using the same paths, but generated
        // twice. This isn't efficient and should be refactored.
        auto mdl_up = mdl->bump(sensitivity, bump_size);
        auto mdl_dn = mdl->bump(sensitivity, -bump_size);
        return (price(opt, sim, mdl_up) - price(opt, sim, mdl_dn)) /
               (2 * bump_size);
    }

   private:
    // Wrapper for double to alleviate effects of false sharing.
    struct aligned_double {
        __host__ aligned_double(double value) : value_(value) {}
        alignas(64) double value_;
    };

    template <typename Option, typename Simulater, typename Model>
    __host__ double price_cpp(std::shared_ptr<Option> opt,
                              std::shared_ptr<Simulater> sim,
                              std::shared_ptr<Model> mdl) {
        std::size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> ts;
        ts.reserve(num_threads);

        std::vector<aligned_double> thread_res(num_threads, 0.0);

        // Each thread gets num_paths_ / num_threads, except the last one which
        // gets num_paths_ / num_threads + num_paths_ % num_threads.
        auto work = [&](std::size_t thread_id) {
            std::size_t sims = num_paths_ / num_threads;
            std::size_t seed = sims * thread_id;
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

        auto sum_of_paths = [&]() {
            double sum = 0.0;
            for (auto& res : thread_res) {
                sum += res.value_;
            }
            return sum;
        }();

        return mdl->discount_factor() * sum_of_paths / num_paths_;
    }

    template <typename Option, typename Simulater, typename Model>
    __host__ double price_cuda(std::shared_ptr<Option> opt,
                               std::shared_ptr<Simulater> sim,
                               std::shared_ptr<Model> mdl) {
        // Copy data onto device.
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

        return mdl->discount_factor() * sum / num_paths_;
    }

    std::size_t num_paths_;
};

}  // namespace emce
