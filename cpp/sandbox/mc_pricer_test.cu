#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <execution>
#include <ranges>
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

}  // namespace

TEST(CUDA, BlackScholesCallCpp) {
    constexpr kcu::mc::dispatch_type type = kcu::mc::dispatch_type::cpp;

    double S_0 = 25.0;
    double r = 0.03;
    double sigma = 0.2;
    double K = 24.9;
    double T = 1.0;

    kcu::mc::black_scholes model(S_0, r, sigma);
    kcu::mc::vanilla_call option(K);
    kcu::mc::euler_maruyama simulater(50, T);

    auto pricer = kcu::mc::monte_carlo_pricer<type>(1e7);
    auto price = pricer.run(option, simulater, model, T);

    EXPECT_TRUE(std::abs(price - analytical_call_price(S_0, K, r, sigma, T)) <
                static_cast<double>(1e-2));
}

TEST(CUDA, BlackScholesCallCUDA) {
    constexpr kcu::mc::dispatch_type type = kcu::mc::dispatch_type::cuda;

    double S_0 = 25.0;
    double r = 0.03;
    double sigma = 0.2;
    double K = 24.9;
    double T = 1.0;

    kcu::mc::black_scholes model(S_0, r, sigma);
    kcu::mc::vanilla_call option(K);
    kcu::mc::euler_maruyama simulater(50, T);

    auto pricer = kcu::mc::monte_carlo_pricer<type>(1e7);
    auto price = pricer.run(option, simulater, model, T);

    EXPECT_TRUE(std::abs(price - analytical_call_price(S_0, K, r, sigma, T)) <
                static_cast<double>(1e-2));
}

TEST(CUDA, BlackScholesPutCpp) {
    constexpr kcu::mc::dispatch_type type = kcu::mc::dispatch_type::cpp;

    double S_0 = 25.0;
    double r = 0.03;
    double sigma = 0.2;
    double K = 24.9;
    double T = 1.0;

    kcu::mc::black_scholes model(S_0, r, sigma);
    kcu::mc::vanilla_put option(K);
    kcu::mc::euler_maruyama simulater(50, T);

    auto pricer = kcu::mc::monte_carlo_pricer<type>(1e7);
    auto price = pricer.run(option, simulater, model, T);

    EXPECT_TRUE(std::abs(price - analytical_put_price(S_0, K, r, sigma, T)) <
                static_cast<double>(1e-1));
}

TEST(CUDA, BlackScholesPutCUDA) {
    constexpr kcu::mc::dispatch_type type = kcu::mc::dispatch_type::cuda;

    double S_0 = 25.0;
    double r = 0.03;
    double sigma = 0.2;
    double K = 24.9;
    double T = 1.0;

    kcu::mc::black_scholes model(S_0, r, sigma);
    kcu::mc::vanilla_put option(K);
    kcu::mc::euler_maruyama simulater(50, T);

    auto pricer = kcu::mc::monte_carlo_pricer<type>(1e7);
    auto price = pricer.run(option, simulater, model, T);

    EXPECT_TRUE(std::abs(price - analytical_put_price(S_0, K, r, sigma, T)) <
                static_cast<double>(1e-2));
}