#include <cuda_runtime.h>
#include "emce/option/down_and_out_call.cuh"

namespace emce {

double down_and_out_call::payoff_impl(double* spots) const {
    std::size_t size = base_t::periods();
    for (std::size_t i = 0; i < size; ++i) {
        if (spots[i] <= barrier_) {
            return 0.0;
        }
    }
    return max(spots[size - 1] - K_, 0.0);
}

}  // namespace emce