#pragma once

#include <cuda_runtime.h>

namespace emce {

template <typename Derived>
class option {
   public:
    template <typename... Args>
    __host__ __device__ double payoff(Args&&... args) const {
        return static_cast<const Derived*>(this)->payoff_impl(
            std::forward<Args>(args)...);
    }
};

}  // namespace emce