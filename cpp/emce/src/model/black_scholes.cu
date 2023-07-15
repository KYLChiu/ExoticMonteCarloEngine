#include "emce/model/black_scholes.cuh"

namespace emce {

double black_scholes::drift_impl(double X_t, double t) const {
    return base_t::parameters_[1] * X_t;
}
double black_scholes::diffusion_impl(double X_t, double t) const {
    return base_t::parameters_[2] * X_t;
}
double black_scholes::discount_factor_impl() const {
    return exp(-base_t::parameters_[1] * base_t::parameters_[3]);
}
double black_scholes::parameter_impl(parameters parameter) const {
    return base_t::parameters_[static_cast<int>(parameter)];
}
std::shared_ptr<black_scholes> black_scholes::bump_impl(
    sensitivities sensitivity, double bump_size) const {
    auto mdl = std::make_shared<black_scholes>(*this);
    std::size_t idx = static_cast<int>(sensitivity);
    double param = mdl->parameter(parameters(idx));
    mdl->base_t::parameters_[idx] = param + bump_size;
    return mdl;
}

}  // namespace emce