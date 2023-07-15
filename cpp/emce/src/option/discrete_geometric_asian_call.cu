#include <cuda_runtime.h>
#include "emce/option/discrete_geometric_asian_call.cuh"

namespace emce {

double discrete_geometric_asian_call::payoff_impl(double* spots) const {
    // TODO: this may benefit from having a separate GPU impl.
    double G = 1.0;
    double size = base_t::periods();
    for (std::size_t i = 0; i < size; ++i) {
        G *= spots[i];
    }
    G = pow(G, 1.0 / size);
    return max(G - K_, 0.0);
}

}  // namespace emce