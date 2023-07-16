#pragma once

#include <cuda_runtime.h>
#include "emce/option/path_dependent_option.cuh"

namespace emce {

// https://en.wikipedia.org/wiki/Barrier_option
class down_and_out_call final
    : public path_dependent_option<down_and_out_call> {
    using base_t = path_dependent_option<down_and_out_call>;
    friend class option<down_and_out_call>;

   public:
    __host__ __device__ explicit down_and_out_call(double K, double barrier)
        : K_(K),
          barrier_(barrier),
          base_t(1e3 /* this needs to be thought through better */) {}

   private:
    __host__ __device__ double payoff_impl(double* spots) const;

    double K_;
    double barrier_;
};

}  // namespace emce