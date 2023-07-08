#pragma once

#include <cuda_runtime.h>

namespace emce {

// Class to hold payoffs. The arguments are variadic to support different
// path dependent versus terminal payoffs.
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