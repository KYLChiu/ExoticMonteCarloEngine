#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <chrono>
#include <cmath>
#include <type_traits>
#include "monte_carlo_pricer/model/black_scholes.cuh"
#include "monte_carlo_pricer/monte_carlo_pricer.cuh"
#include "monte_carlo_pricer/option/discrete_geometric_asian_call.cuh"
#include "monte_carlo_pricer/option/down_and_out_call.cuh"
#include "monte_carlo_pricer/option/european_call.cuh"
#include "monte_carlo_pricer/option/european_put.cuh"
#include "monte_carlo_pricer/simulater/analytical_simulater.cuh"
#include "monte_carlo_pricer/simulater/euler_maruyama.cuh"

namespace {

double d_j(double j, double S, double K, double r, double v, double T) {
    return (log(S / K) + (r + (pow(-1, j - 1)) * 0.5 * v * v) * T) /
           (v * sqrt(T));
}

double norm_cdf(double value) { return 0.5 * erfc(-value * M_SQRT1_2); }

double analytical_euro_call_price(double S, double K, double r, double v,
                                  double T) {
    return S * norm_cdf(d_j(1, S, K, r, v, T)) -
           K * exp(-r * T) * norm_cdf(d_j(2, S, K, r, v, T));
}

double analytical_euro_put_price(double S, double K, double r, double v,
                                 double T) {
    return -S * norm_cdf(-d_j(1, S, K, r, v, T)) +
           K * exp(-r * T) * norm_cdf(-d_j(2, S, K, r, v, T));
}

double analytical_geom_asian_call_price(double S, double K, double r, double v,
                                        double T, std::size_t periods) {
    auto sigma = sqrt((v * v * (periods + 1) * (2 * periods + 1)) /
                      (6 * periods * periods));
    auto mu =
        (r - 0.5 * v * v) * (periods + 1) / (2 * periods) + 0.5 * sigma * sigma;
    auto [d_1, d_2] =
        std::make_pair(d_j(1, S, K, mu, sigma, T), d_j(2, S, K, mu, sigma, T));
    return exp(-r * T) * (S * exp(mu * T) * norm_cdf(d_1) - K * norm_cdf(d_2));
}

double analytical_down_and_out_call_price(double S, double K, double r,
                                          double v, double T, double barrier) {
    double alpha = -r / (v * v) + 0.5;
    return analytical_euro_call_price(S, K, r, v, T) -
           pow(barrier / S, 2 * alpha) *
               analytical_euro_call_price(barrier * barrier / S, K, r, v, T);
}

template <emce::dispatch_type DispatchType, typename Option, typename Simulater>
void run_test(std::size_t num_steps = 1e2, std::size_t num_paths = 1e6,
              std::size_t num_options = 1) {
    using namespace emce;
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    double S_0 = 40.0;
    double r = 0.03;
    double sigma = 0.15;
    double K = 40.0;
    double T = 0.5;

    std::size_t periods = 15;   // for Asian option
    double barrier = 0.90 * K;  // for down and out barrier

    double tol = 1e-2;

    std::size_t total_elapsed = 0;

    auto pricer = monte_carlo_pricer<DispatchType>(num_paths);

    for (std::size_t i = 0; i < num_options; ++i) {
        auto option = [&]() -> std::shared_ptr<Option> {
            if constexpr (std::is_same_v<Option, european_call>) {
                return std::make_shared<european_call>(K);
            }
            if constexpr (std::is_same_v<Option, european_put>) {
                return std::make_shared<european_put>(K);
            }
            if constexpr (std::is_same_v<Option,
                                         discrete_geometric_asian_call>) {
                return std::make_shared<discrete_geometric_asian_call>(K,
                                                                       periods);
            }
            if constexpr (std::is_same_v<Option, down_and_out_call>) {
                return std::make_shared<down_and_out_call>(K, barrier);
            }
            throw std::runtime_error("Unexpected option type.");
        }();

        auto simulater = [&]() {
            if constexpr (std::is_same_v<Simulater, euler_maruyama>) {
                return std::make_shared<euler_maruyama>(num_steps);
            } else {
                return std::make_shared<analytical_simulater>();
            }
        }();

        auto model = std::make_shared<black_scholes>(S_0, r, sigma, T);

        auto analytical_price = [&]() -> double {
            if constexpr (std::is_same_v<Option, european_call>) {
                return analytical_euro_call_price(S_0, K, r, sigma, T);
            }
            if constexpr (std::is_same_v<Option, european_put>) {
                return analytical_euro_put_price(S_0, K, r, sigma, T);
            }
            if constexpr (std::is_same_v<Option,
                                         discrete_geometric_asian_call>) {
                return analytical_geom_asian_call_price(S_0, K, r, sigma, T,
                                                        periods);
            }
            if constexpr (std::is_same_v<Option, down_and_out_call>) {
                return analytical_down_and_out_call_price(S_0, K, r, sigma, T,
                                                          barrier);
            }
            throw std::runtime_error("Unexpected option type.");
        }();

        auto start = high_resolution_clock::now();
        double price = pricer.price(option, simulater, model);
        auto end = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(end - start);
        total_elapsed += elapsed.count();

        double abs_err = std::abs(price - analytical_price);
        double rel_err = std::abs(abs_err / analytical_price);
        std::cout << "MC Price         : " << price << "\n";
        std::cout << "Analytical Price : " << analytical_price << "\n";
        std::cout << "Absolute Error   : " << abs_err << "\n";
        std::cout << "Relative Error   : " << rel_err << "\n";
        EXPECT_TRUE(rel_err < tol);

        S_0++;
        K++;
        T += 0.1;
        sigma += 0.01;
        barrier -= 0.01;
    }

    std::cout << "Average elapsed time per option (ms): "
              << total_elapsed / num_options << "\n";
}

}  // namespace

TEST(EMCE, Cpp_EuropeanCall_Analytic_BS) {
    run_test<emce::dispatch_type::cpp, emce::european_call,
             emce::analytical_simulater>();
}

TEST(EMCE, CUDA_EuropeanCall_Analytic_BS) {
    run_test<emce::dispatch_type::cuda, emce::european_call,
             emce::analytical_simulater>();
}

TEST(EMCE, Cpp_EuropeanCall_EM_BS) {
    run_test<emce::dispatch_type::cpp, emce::european_call,
             emce::euler_maruyama>();
}

TEST(EMCE, CUDA_EuropeanCall_EM_BS) {
    run_test<emce::dispatch_type::cpp, emce::european_call,
             emce::euler_maruyama>();
}

TEST(EMCE, Cpp_Put_EuropeanPut_Analytic_BS) {
    run_test<emce::dispatch_type::cpp, emce::european_put,
             emce::analytical_simulater>();
}

TEST(EMCE, CUDA_EuropeanPut_Analytic_BS) {
    run_test<emce::dispatch_type::cuda, emce::european_put,
             emce::analytical_simulater>();
}

TEST(EMCE, Cpp_EuropeanPut_EM_BS) {
    run_test<emce::dispatch_type::cpp, emce::european_put,
             emce::euler_maruyama>();
}

TEST(EMCE, CUDA_EuropeanPut_EM_BS) {
    run_test<emce::dispatch_type::cuda, emce::european_put,
             emce::euler_maruyama>();
}

TEST(EMCE, Cpp_GeometricAsianCall_EM_BS) {
    run_test<emce::dispatch_type::cpp, emce::discrete_geometric_asian_call,
             emce::euler_maruyama>();
}

TEST(EMCE, CUDA_GeometricAsianCall_EM_BS) {
    run_test<emce::dispatch_type::cuda, emce::discrete_geometric_asian_call,
             emce::euler_maruyama>();
}

TEST(EMCE, Cpp_DownAndOutCall_EM_BS) {
    run_test<emce::dispatch_type::cpp, emce::down_and_out_call,
             emce::euler_maruyama>();
}

TEST(EMCE, CUDA_DownAndOutCall_EM_BS) {
    run_test<emce::dispatch_type::cuda, emce::down_and_out_call,
             emce::euler_maruyama>();
}