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

double norm_pdf(double x) {
    return (1.0 / (pow(2 * M_PI, 0.5))) * exp(-0.5 * x * x);
}

double norm_cdf(double value) { return 0.5 * erfc(-value * M_SQRT1_2); }

double bs_euro_call_delta(double S, double K, double r, double v, double T) {
    return norm_cdf(d_j(1, S, K, r, v, T));
}

double bs_euro_call_vega(double S, double K, double r, double v, double T) {
    return S * norm_pdf(d_j(1, S, K, r, v, T)) * sqrt(T);
}

double bs_euro_call_theta(double S, double K, double r, double v, double T) {
    return (S * norm_pdf(d_j(1, S, K, r, v, T)) * v) / (2 * sqrt(T)) +
           r * K * exp(-r * T) * norm_cdf(d_j(2, S, K, r, v, T));
}

double bs_euro_call_rho(double S, double K, double r, double v, double T) {
    return K * T * exp(-r * T) * norm_cdf(d_j(2, S, K, r, v, T));
}

// Greeks follow BS model exactly - unit tests would break if incorporating
// dividend/repo rate
template <emce::dispatch_type DispatchType, typename Option>
void run_test(emce::black_scholes::sensitivities sensitivity,
              double bump_size = 1e-6, std::size_t num_steps = 1e2,
              std::size_t num_paths = 1e6, std::size_t num_options = 1) {
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
    double tol = 1e-2;

    std::size_t total_elapsed = 0;

    auto pricer = monte_carlo_pricer<DispatchType>(num_paths);

    for (std::size_t i = 0; i < num_options; ++i) {
        auto option = [&]() -> std::shared_ptr<Option> {
            if constexpr (std::is_same_v<Option, european_call>) {
                return std::make_shared<european_call>(K);
            }
            throw std::runtime_error("Unexpected option type.");
        }();

        auto simulater = [&]() {
            return std::make_shared<analytical_simulater>();
        }();

        auto model = std::make_shared<black_scholes>(S_0, r, sigma, T);

        auto analytical_greek = [&]() -> double {
            if (sensitivity == black_scholes::sensitivities::delta) {
                return bs_euro_call_delta(S_0, K, r, sigma, T);
            }
            if (sensitivity == black_scholes::sensitivities::vega) {
                return bs_euro_call_vega(S_0, K, r, sigma, T);
            }
            if (sensitivity == black_scholes::sensitivities::theta) {
                return bs_euro_call_theta(S_0, K, r, sigma, T);
            }
            if (sensitivity == black_scholes::sensitivities::rho) {
                return bs_euro_call_rho(S_0, K, r, sigma, T);
            }
            throw std::runtime_error("Unexpected sensitivity type.");
        }();

        auto start = high_resolution_clock::now();
        double greek = pricer.sensitivity(option, simulater, model, sensitivity,
                                          bump_size);
        auto end = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(end - start);
        total_elapsed += elapsed.count();

        double abs_err = std::abs(greek - analytical_greek);
        double rel_err = std::abs(abs_err / analytical_greek);
        std::cout << "MC Greek         : " << greek << "\n";
        std::cout << "Analytical Greek : " << analytical_greek << "\n";
        std::cout << "Absolute Error   : " << abs_err << "\n";
        std::cout << "Relative Error   : " << rel_err << "\n";
        EXPECT_TRUE(rel_err < tol);

        S_0++;
        K++;
        T += 0.1;
        sigma += 0.01;
    }

    std::cout << "Average elapsed time per option (ms): "
              << total_elapsed / num_options << "\n";
}

}  // namespace

TEST(EMCE, Cpp_EuropeanCall_Delta_BS) {
    run_test<emce::dispatch_type::cpp, emce::european_call>(
        emce::black_scholes::sensitivities::delta);
}

TEST(EMCE, CUDA_EuropeanCall_Delta_BS) {
    run_test<emce::dispatch_type::cuda, emce::european_call>(
        emce::black_scholes::sensitivities::delta);
}

TEST(EMCE, Cpp_EuropeanCall_Vega_BS) {
    run_test<emce::dispatch_type::cpp, emce::european_call>(
        emce::black_scholes::sensitivities::vega);
}

TEST(EMCE, CUDA_EuropeanCall_Vega_BS) {
    run_test<emce::dispatch_type::cuda, emce::european_call>(
        emce::black_scholes::sensitivities::vega);
}

TEST(EMCE, Cpp_EuropeanCall_Theta_BS) {
    run_test<emce::dispatch_type::cpp, emce::european_call>(
        emce::black_scholes::sensitivities::theta, 1e-5);
}

TEST(EMCE, CUDA_EuropeanCall_Theta_BS) {
    run_test<emce::dispatch_type::cuda, emce::european_call>(
        emce::black_scholes::sensitivities::theta, 1e-5);
}

TEST(EMCE, Cpp_EuropeanCall_Rho_BS) {
    run_test<emce::dispatch_type::cpp, emce::european_call>(
        emce::black_scholes::sensitivities::rho);
}

TEST(EMCE, CUDA_EuropeanCall_Rho_BS) {
    run_test<emce::dispatch_type::cuda, emce::european_call>(
        emce::black_scholes::sensitivities::rho);
}
