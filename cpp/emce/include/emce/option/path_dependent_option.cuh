#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include "emce/option/option.cuh"

namespace emce {

template <typename Derived>
class path_dependent_option : public option<Derived> {
    friend class option<Derived>;

   public:
    // Number of monitoring periods (Asian options)
    // Number of spots to check against the barrier (barriers)
    __host__ __device__ std::size_t periods() const { return periods_; }

   protected:
    __host__ __device__ explicit path_dependent_option(std::size_t periods)
        : periods_(periods) {
        assert(periods_ > 0);
    }

   private:
    std::size_t periods_;
};

}  // namespace emce