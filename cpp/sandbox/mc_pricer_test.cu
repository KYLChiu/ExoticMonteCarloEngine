#include <gtest/gtest.h>
#include <stdio.h>
#include <cmath>
#include <type_traits>
#include "kc_utils/cuda/monte_carlo_pricer/model/black_scholes.cuh"
#include "kc_utils/cuda/monte_carlo_pricer/monte_carlo_pricer.cuh"
#include "kc_utils/cuda/monte_carlo_pricer/option/vanilla_call.cuh"
#include "kc_utils/cuda/monte_carlo_pricer/option/vanilla_put.cuh"
#include "kc_utils/cuda/monte_carlo_pricer/simulater/euler_maruyama.cuh"

namespace {

double d_j(double j, double S, double K, double r, double v, double T) {
    return (log(S / K) + (r + (pow(-1, j - 1)) * 0.5 * v * v) * T) /
           (v * (pow(T, 0.5)));
}

double norm_cdf(double value) { return 0.5 * erfc(-value * M_SQRT1_2); }

double analytical_call_price(double S, double K, double r, double v, double T) {
    return S * norm_cdf(d_j(1, S, K, r, v, T)) -
           K * exp(-r * T) * norm_cdf(d_j(2, S, K, r, v, T));
}

double analytical_put_price(double S, double K, double r, double v, double T) {
    return -S * norm_cdf(-d_j(1, S, K, r, v, T)) +
           K * exp(-r * T) * norm_cdf(-d_j(2, S, K, r, v, T));
}

template <kcu::mc::dispatch_type DispatchType, typename Option>
void run_test() {
    using namespace kcu::mc;

    double S_0 = 50.0;
    double r = 0.05;
    double sigma = 0.15;
    double K = 50.0;
    double T = 1.0;

    std::size_t num_steps = 5e1;
    std::size_t num_paths = 1e7;

    auto model = std::make_shared<black_scholes>(S_0, r, sigma);
    auto simulater = std::make_shared<euler_maruyama>(num_steps, T);
    auto pricer = monte_carlo_pricer<DispatchType>(num_paths);

    if constexpr (std::is_same_v<Option, vanilla_call>) {
        auto option = std::make_shared<vanilla_call>(K);
        auto price = pricer.run(option, simulater, model, T);
        EXPECT_TRUE(
            std::abs(price - analytical_call_price(S_0, K, r, sigma, T)) <
            static_cast<double>(1e-2));
    } else {
        auto option = std::make_shared<vanilla_put>(K);
        auto price = pricer.run(option, simulater, model, T);
        EXPECT_TRUE(
            std::abs(price - analytical_put_price(S_0, K, r, sigma, T)) <
            static_cast<double>(1e-2));
    }
}

}  // namespace

TEST(CUDA, BlackScholesCallCpp) {
    run_test<kcu::mc::dispatch_type::cpp, kcu::mc::vanilla_call>();
}

TEST(CUDA, BlackScholesCallCUDA) {
    run_test<kcu::mc::dispatch_type::cuda, kcu::mc::vanilla_call>();
}

TEST(CUDA, BlackScholesPutCpp) {
    run_test<kcu::mc::dispatch_type::cpp, kcu::mc::vanilla_put>();
}

TEST(CUDA, BlackScholesPutCUDA) {
    run_test<kcu::mc::dispatch_type::cuda, kcu::mc::vanilla_put>();
}