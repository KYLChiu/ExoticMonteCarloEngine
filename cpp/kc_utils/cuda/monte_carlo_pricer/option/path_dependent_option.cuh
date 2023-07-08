#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include "option.cuh"

namespace kcu::mc {

template <typename Derived>
class path_dependent_option : public option<Derived> {
    friend class option<Derived>;

   public:
    __host__ __device__ constexpr std::size_t periods() const {
        return periods_;
    }

   protected:
    __host__ __device__ path_dependent_option(std::size_t periods)
        : periods_(periods) {
        assert(periods_ > 0);
    }

   private:
    std::size_t periods_;
};

}  // namespace kcu::mc