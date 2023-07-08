#pragma once

#include <cuda_runtime.h>
#include "option.cuh"
#include "path_dependent_option.cuh"

namespace emce {

// https://quant.stackexchange.com/questions/68929/discrete-geometric-asian-option-call-price-formula
class discrete_geometric_asian_call final
    : public path_dependent_option<discrete_geometric_asian_call> {
    using base_t = path_dependent_option<discrete_geometric_asian_call>;
    friend class option<discrete_geometric_asian_call>;

   public:
    __host__ __device__ discrete_geometric_asian_call(double K,
                                                      std::size_t periods)
        : K_(K), base_t(periods) {}

   private:
    __host__ __device__ double payoff_impl(double* spots) const {
        // TODO: this may benefit from having a separate GPU impl.
        double G = 1.0;
        double size = base_t::periods();
        for (std::size_t i = 0; i < size; ++i) {
            G *= spots[i];
        }
        G = pow(G, 1.0 / size);
        return max(G - K_, 0.0);
    }

    double K_;
};

}  // namespace emce