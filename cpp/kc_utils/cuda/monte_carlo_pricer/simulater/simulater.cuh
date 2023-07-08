#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <memory>

namespace kcu::mc {

template <typename Derived>
class simulater {
   public:
    template <typename Option, typename Model>
    __host__ double simulate_cpp(std::shared_ptr<Option> option,
                                 std::shared_ptr<Model> model,
                                 std::size_t rng_seed) const {
        return static_cast<const Derived*>(this)->simulate_cpp_impl(
            option, model, rng_seed);
    }

    template <typename Option, typename Model>
    __device__ double simulate_cuda(thrust::device_ptr<Option> option,
                                    thrust::device_ptr<Model> model,
                                    std::size_t rng_seed) const {
        return static_cast<const Derived*>(this)->simulate_cuda_impl(
            option, model, rng_seed);
    }
};

}  // namespace kcu::mc
