#pragma once

#include <cuda_runtime.h>

namespace kcu::mc {

/* -- OPTION --
Represents an option. Requires a payoff.
*/

template <typename Derived>
class option {
   public:
    __host__ __device__ double payoff(double S) const {
        return static_cast<const Derived*>(this)->payoff_impl(S);
    }
};
}  // namespace kcu::mc