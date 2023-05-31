#pragma once

#include <cuda_runtime.h>
#include <execution>
#include <functional>
#include <future>
#include <memory>
#include <random>
#include <thread>

namespace kcu
{

    /* Represents dX_t = drift_t dt + diffusion_t dW_t, where drift_t and diffusion_t are deterministic and dW_t is a Brownian motion. */
    class first_order_sde
    {
    public:
        using drift_t = double;
        using diffusion_t = double;
        using value_t = double;
        using time_t = double;
        using coefficient_generator_t = std::function<std::pair<drift_t, diffusion_t>(value_t, time_t)>;

        virtual ~first_order_sde() = default;

        virtual std::pair<drift_t, diffusion_t> coefficients(value_t X_tminus1, time_t tminus1) const
        {
            return cg_(X_tminus1, tminus1);
        }

        auto initial_value() const { return X_0_; };

    protected:
        explicit first_order_sde(double X_0, coefficient_generator_t &&cg) : X_0_(X_0), cg_(std::move(cg)) {}
        double X_0_;
        coefficient_generator_t cg_;
    };

    class geometric_brownian_motion final : public first_order_sde
    {
    public:
        explicit geometric_brownian_motion(double X_0, coefficient_generator_t &&cg) : first_order_sde(X_0, std::move(cg)) {}
    };

    class first_order_sde_simulater
    {
    public:
        virtual ~first_order_sde_simulater() = default;
        virtual double simulate(const first_order_sde &sde, double T) const = 0;
    };

    template <std::size_t Steps>
    class euler_maryuama : public first_order_sde_simulater
    {
    public:
        virtual double simulate(const first_order_sde &sde, double T) const override
        {
            double dt = T / Steps;

            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<double> dist(0, std::sqrt(dt));

            double X_t = sde.initial_value();
            double t = 0.0f;
            for (std::size_t i = 0; i < Steps; ++i)
            {
                auto [drift, diffusion] = sde.coefficients(X_t, t);
                double dWt = dist(gen);
                X_t = X_t + drift * dt + diffusion * dWt;
                t += dt;
            }
            return X_t;
        }
    };

    template <std::size_t NumPaths, typename Payoff>
    class monte_carlo_engine final
    {
    public:
        inline static double run(const Payoff &payoff, const first_order_sde_simulater &simulater, const first_order_sde &sde, double T)
        {
            std::vector<std::thread> ts;
            auto num_threads = std::thread::hardware_concurrency();
            ts.reserve(num_threads);

            std::vector<std::size_t> num_sims_per_thread(num_threads, NumPaths / num_threads);
            num_sims_per_thread[0] += NumPaths % num_threads;

            std::vector<double> thread_res(num_threads, 0.0f);

            auto f = [&](std::size_t thread_id)
            {
                for (std::size_t i = 0; i < num_sims_per_thread[thread_id]; ++i)
                {
                    thread_res[thread_id] += payoff(simulater.simulate(sde, T));
                }
            };

            for (std::size_t i = 0; i < num_threads; ++i)
            {
                ts.emplace_back(std::thread(f, i));
            }

            for (auto &t : ts)
            {
                t.join();
            }

            double sum = 0;
            for (const auto &res : thread_res)
            {
                sum += res;
            }

            return sum /= NumPaths;
        }
    };

} // namespace kcu
