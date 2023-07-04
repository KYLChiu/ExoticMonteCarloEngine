#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace kcu::mc {

template <typename Derived>
class simulater {
   public:
    template <typename Model>
    __host__ double simulate_cpp(std::shared_ptr<Model> model,
                                 std::size_t rng_seed) const {
        return static_cast<const Derived*>(this)->simulate_cpp_impl(model,
                                                                    rng_seed);
    }

    template <typename Model>
    __device__ double simulate_cuda(thrust::device_ptr<Model> model,
                                    std::size_t rng_seed) const {
        return static_cast<const Derived*>(this)->simulate_cuda_impl(model,
                                                                     rng_seed);
    }
};

}  // namespace kcu::mc
