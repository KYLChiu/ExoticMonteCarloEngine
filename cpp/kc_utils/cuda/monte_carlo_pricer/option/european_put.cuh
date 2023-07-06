#pragma once

#include <cuda_runtime.h>
#include "option.cuh"

namespace kcu::mc {

class european_put : public option<european_put> {
    friend class option<european_put>;

   public:
    __host__ __device__ european_put(double K) : K_(K) {}

   private:
    __host__ __device__ double payoff_impl(double S) const {
        return max(K_ - S, 0.0);
    }

    double K_;
};

}  // namespace kcu::mc
