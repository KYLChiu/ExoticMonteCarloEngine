#include <gtest/gtest.h>
#include <stdio.h>
#include <cmath>
#include <type_traits>
#include "kc_utils/cuda/monte_carlo_pricer/model/black_scholes.cuh"
#include "kc_utils/cuda/monte_carlo_pricer/monte_carlo_pricer.cuh"
#include "kc_utils/cuda/monte_carlo_pricer/option/vanilla_call.cuh"
#include "kc_utils/cuda/monte_carlo_pricer/option/vanilla_put.cuh"
#include "kc_utils/cuda/monte_carlo_pricer/simulater/analytical_simulater.cuh"
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

template <kcu::mc::dispatch_type DispatchType, typename Option,
          typename Simulater>
void run_test(std::size_t num_steps, std::size_t num_paths) {
    using namespace kcu::mc;

    double S_0 = 25.0;
    double r = 0.03;
    double sigma = 0.15;
    double K = 25.0;
    double T = 0.5;

    double tol = 1e-3;

    auto model = std::make_shared<black_scholes>(S_0, r, sigma);
    auto pricer = monte_carlo_pricer<DispatchType>(num_paths);

    auto option = [&]() {
        if constexpr (std::is_same_v<Option, vanilla_call>) {
            return std::make_shared<vanilla_call>(K);
        } else {
            return std::make_shared<vanilla_put>(K);
        }
    }();

    auto simulater = [&]() {
        if constexpr (std::is_same_v<Simulater, euler_maruyama>) {
            return std::make_shared<euler_maruyama>(num_steps, T);
        } else {
            return std::make_shared<analytical_simulater>(T);
        }
    }();

    auto analytical_price = [&] {
        if constexpr (std::is_same_v<Option, vanilla_call>) {
            return analytical_call_price(S_0, K, r, sigma, T);
        } else {
            return analytical_put_price(S_0, K, r, sigma, T);
        }
    }();

    double price = pricer.run(option, simulater, model, T);
    double diff = std::abs(price - analytical_price);
    std::cout << "MC Price         : " << price << std::endl;
    std::cout << "Analytical Price : " << analytical_price << std::endl;
    std::cout << "Difference       : " << diff << std::endl;
    EXPECT_TRUE(diff < tol);
}

}  // namespace

TEST(MonteCarlo, Cpp_Put_EM_BS) {
    run_test<kcu::mc::dispatch_type::cpp, kcu::mc::vanilla_put,
             kcu::mc::euler_maruyama>(1e1, 1e8);
}

TEST(MonteCarlo, CUDA_Put_EM_BS) {
    run_test<kcu::mc::dispatch_type::cuda, kcu::mc::vanilla_put,
             kcu::mc::euler_maruyama>(1e1, 1e8);
}

TEST(MonteCarlo, Cpp_Call_EM_BS) {
    run_test<kcu::mc::dispatch_type::cpp, kcu::mc::vanilla_call,
             kcu::mc::euler_maruyama>(1e1, 1e8);
}

TEST(MonteCarlo, CUDA_Call_EM_BS) {
    run_test<kcu::mc::dispatch_type::cpp, kcu::mc::vanilla_call,
             kcu::mc::euler_maruyama>(1e1, 1e8);
}

TEST(MonteCarlo, Cpp_Call_Analytic_BS) {
    run_test<kcu::mc::dispatch_type::cpp, kcu::mc::vanilla_call,
             kcu::mc::analytical_simulater>(0, 1e9);
}

TEST(MonteCarlo, CUDA_Call_Analytic_BS) {
    run_test<kcu::mc::dispatch_type::cuda, kcu::mc::vanilla_call,
             kcu::mc::analytical_simulater>(0, 1e9);
}
