#pragma once

#include <cuda_runtime.h>
#include <type_traits>

namespace emce {

// Representation of the model dynamics: first order SDE dX_t = f(X_t, t) dt +
// g(X_t, t) dW_t, where the drift (f) and diffusion (g) are deterministic
// functions and dW_t is a Brownian motion. Needs also to provide a discount
// factor function.
template <typename Derived, std::size_t NumParameters>
class model {
   public:
    template <typename... Args>
    __host__ __device__ model(Args&&... args)
        : parameters_{static_cast<double>(args)...} {
        static_assert(sizeof...(args) == NumParameters,
                      "Number of arguments does not match the expected number "
                      "of model parameters.");
    }

    __host__ __device__ model(const model& other) {
        for (std::size_t i = 0; i < NumParameters; ++i) {
            parameters_[i] = other.parameters_[i];
        }
    }

    __host__ __device__ model& operator=(const model& other) {
        if (this != &other) {
            parameters_ = other.parameters_;
        }
        return *this;
    }

    // No move ctor/assignment operator as no std::move() for device.

    __host__ __device__ virtual ~model() {}

    __host__ __device__ double drift(double X_t, double t) const {
        return static_cast<const Derived*>(this)->drift_impl(X_t, t);
    }

    __host__ __device__ double diffusion(double X_t, double t) const {
        return static_cast<const Derived*>(this)->diffusion_impl(X_t, t);
    }

    __host__ __device__ double discount_factor() const {
        return static_cast<const Derived*>(this)->discount_factor_impl();
    }

    // Returns the parameter given an enum in the derived class.
    // TODO: rethink how this should work. The enum class should be defined by
    // the derived class but it is not clear for users/developers.
    template <typename Enum>
    __host__ __device__ double parameter(Enum parameter) const {
        return static_cast<const Derived*>(this)->parameter_impl(parameter);
    }

    // Returns a bumped model where the model has an absolute bump_size bump to
    // the parameter corresponding to the input sensitivity. This is needed for
    // Greeks.
    // TODO: rethink how this should work. The enum class should be defined by
    // the derived class but it is not clear for users/developers.
    template <typename Enum>
    __host__ std::shared_ptr<Derived> bump(Enum sensitivity,
                                           double bump_size) const {
        return static_cast<const Derived*>(this)->bump_impl(sensitivity,
                                                            bump_size);
    }

   protected:
    double parameters_[NumParameters];
};

}  // namespace emce
