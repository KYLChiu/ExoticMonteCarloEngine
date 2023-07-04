#pragma once

#include <cuda_runtime.h>

namespace kcu::mc {

template <typename Derived>
class simulater {
   public:
    template <typename Model>
    __host__ double simulate_cpp(const Model& model,
                                 std::size_t rng_seed) const {
        return static_cast<const Derived*>(this)->simulate_cpp_impl(model,
                                                                    rng_seed);
    }

    template <typename ModelPtr>
    __device__ double simulate_cuda(ModelPtr model,
                                    std::size_t rng_seed) const {
        return static_cast<const Derived*>(this)->simulate_cuda_impl(model,
                                                                     rng_seed);
    }
};

}  // namespace kcu::mc
