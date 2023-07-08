#pragma once

#include <cuda_runtime.h>
#include "model.cuh"

namespace emce {

class black_scholes final : public model<black_scholes> {
    friend class model<black_scholes>;

   public:
    // dS_t = rS_t dt + sigma S_t dW_t
    // https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    __host__ __device__ black_scholes(double S_0 /*initial value*/,
                                      double r /*risk-free rate*/,
                                      double sigma /*volatility*/)
        : S_0_(S_0), r_(r), sigma_(sigma) {}

    __host__ __device__ double r() const { return r_; }
    __host__ __device__ double sigma() const { return sigma_; }

   private:
    __host__ __device__ double initial_value_impl() const { return S_0_; }
    __host__ __device__ double drift_impl(double X_t, double t) const {
        return r_ * X_t;
    }
    __host__ __device__ double diffusion_impl(double X_t, double t) const {
        return sigma_ * X_t;
    }
    __host__ __device__ double discount_factor_impl(double t) const {
        return exp(-r_ * t);
    }

   private:
    double S_0_;
    double r_;
    double sigma_;
};

}  // namespace emce