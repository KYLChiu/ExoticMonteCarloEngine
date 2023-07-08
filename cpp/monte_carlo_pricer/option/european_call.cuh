#pragma once

#include <cuda_runtime.h>
#include "option.cuh"

namespace emce {

// https://en.wikipedia.org/wiki/Call_option
class european_call : public option<european_call> {
    friend class option<european_call>;

   public:
    __host__ __device__ european_call(double K) : K_(K) {}

   private:
    __host__ __device__ double payoff_impl(double S) const {
        return max(S - K_, 0.0);
    }

    double K_;
};

}  // namespace emce
