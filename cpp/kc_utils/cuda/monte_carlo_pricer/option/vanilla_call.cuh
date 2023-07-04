#pragma once

#include <cuda_runtime.h>
#include "option.cuh"

namespace kcu::mc {

class vanilla_call : public option<vanilla_call> {
   public:
    __host__ __device__ vanilla_call(double K) : K_(K) {}

    __host__ __device__ double payoff_impl(double S) const {
        return max(S - K_, 0.0);
    }

   private:
    double K_;
};

}  // namespace kcu::mc
