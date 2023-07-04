#pragma once

#include <cuda_runtime.h>
#include "option.cuh"

namespace kcu::mc {

class vanilla_put : public option<vanilla_put> {
   public:
    __host__ __device__ vanilla_put(double K) : K_(K) {}

    __host__ __device__ double payoff_impl(double S) const {
        return max(K_ - S, 0.0);
    }

   private:
    double K_;
};

}  // namespace kcu::mc
