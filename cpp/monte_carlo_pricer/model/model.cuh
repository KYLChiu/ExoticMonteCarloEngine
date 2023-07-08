#pragma once

#include <cuda_runtime.h>

namespace emce {

// Representation of the model dynamics: first order SDE dX_t = f(X_t, t) dt +
// g(X_t, t) dW_t, where the drift (f) and diffusion (g) are deterministic
// functions and dW_t is a Brownian motion. Needs also to provide a discount
// factor function.
template <typename Derived>
class model {
   public:
    __host__ __device__ double initial_value() const {
        return static_cast<const Derived*>(this)->initial_value_impl();
    }

    __host__ __device__ double drift(double X_t, double t) const {
        return static_cast<const Derived*>(this)->drift_impl(X_t, t);
    }

    __host__ __device__ double diffusion(double X_t, double t) const {
        return static_cast<const Derived*>(this)->diffusion_impl(X_t, t);
    }

    __host__ __device__ double discount_factor(double t) const {
        return static_cast<const Derived*>(this)->discount_factor_impl(t);
    }
};

}  // namespace emce
