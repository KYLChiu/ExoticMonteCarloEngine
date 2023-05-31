#include "kc_utils/cuda/first_order_sde.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <execution>
#include <gtest/gtest.h>
#include <ranges>
#include <stdio.h>

namespace
{
    __global__ void cudakernel(double *buf, size_t M, size_t N)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        buf[i] = 1.0f * i / N;
        for (int j = 0; j < M; j++)
            buf[i] = buf[i] * buf[i] - 0.25f;
    }

    double d_j(double j, double S, double K, double r, double v, double T)
    {
        return (log(S / K) + (r + (pow(-1, j - 1)) * 0.5 * v * v) * T) /
               (v * (pow(T, 0.5)));
    }

    double norm_cdf(double value) { return 0.5 * erfc(-value * M_SQRT1_2); }

    double analytical_call_price(double S, double K, double r, double v, double T)
    {
        return S * norm_cdf(d_j(1, S, K, r, v, T)) -
               K * exp(-r * T) * norm_cdf(d_j(2, S, K, r, v, T));
    }

    double analytical_put_price(double S, double K, double r, double v, double T)
    {
        return -S * norm_cdf(-d_j(1, S, K, r, v, T)) +
               K * exp(-r * T) * norm_cdf(-d_j(2, S, K, r, v, T));
    }

} // namespace

TEST(CUDA, BasicCuda)
{
    constexpr size_t N = 512 * 512;
    constexpr size_t M = 100000;
    double data[N];
    double *d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudakernel<<<N / 256, 256>>>(d_data, M, N);
    cudaMemcpy(data, d_data, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    EXPECT_TRUE(std::abs(data[1] - (-0.207107)) < 1e-5);
}

TEST(CUDA, BasicCpp)
{
    constexpr size_t N = 512 * 512;
    constexpr size_t M = 10000;
    double data[N];
    for (int i = 0; i < N; i++)
    {
        data[i] = 1.0 * i / N;
        for (int j = 0; j < M; j++)
        {
            data[i] = data[i] * data[i] - 0.25;
        }
    }
    EXPECT_TRUE(std::abs(data[1] - (-0.207107)) < 1e-5);
}

TEST(CUDA, BlackScholesCallCpp)
{
    double S_0 = 25.0;
    double K = 25.0;
    double r = 0.03;
    double sigma = 0.2;
    double T = 1.0;

    kcu::geometric_brownian_motion<kcu::run_type::cpp> sde(S_0, [&r, &sigma](double X_t, double t)
                                                           { return std::make_pair(r * X_t, sigma * X_t); });

    kcu::euler_maruyama<kcu::run_type::cpp> simulater(100);

    const auto &payoff = [K](double S)
    { return std::max(S - K, 0.0); };

    auto price = std::exp(-r * T) *
                 kcu::monte_carlo_engine<kcu::run_type::cpp>(1000000).run(
                     payoff, simulater, sde, T);

    EXPECT_TRUE(std::abs(price - analytical_call_price(S_0, K, r, sigma, T)) <
                static_cast<double>(1e-2));
}

TEST(CUDA, BlackScholesPutCpp)
{
    double S_0 = 25.0;
    double K = 25.0;
    double r = 0.03;
    double sigma = 0.2;
    double T = 1.0;

    kcu::geometric_brownian_motion<kcu::run_type::cpp> sde(S_0, [&r, &sigma](double X_t, double t)
                                                           { return std::make_pair(r * X_t, sigma * X_t); });

    kcu::euler_maruyama<kcu::run_type::cpp> simulater(100);

    const auto &payoff = [K](double S)
    { return std::max(K - S, 0.0); };

    auto price = std::exp(-r * T) *
                 kcu::monte_carlo_engine<kcu::run_type::cpp>(1000000).run(
                     payoff, simulater, sde, T);

    EXPECT_TRUE(std::abs(price - analytical_put_price(S_0, K, r, sigma, T)) <
                static_cast<double>(1e-2));
}