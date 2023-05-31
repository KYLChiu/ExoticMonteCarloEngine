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

    enum class run_type
    {
        cpp,
        cuda
    };

    /* Represents dX_t = drift_t dt + diffusion_t dW_t, where drift_t and diffusion_t are deterministic and dW_t is a Brownian motion. */
    template <run_type RunType>
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

    template <run_type RunType>
    class geometric_brownian_motion final : public first_order_sde<RunType>
    {
    public:
        explicit geometric_brownian_motion(double X_0, std::function<std::pair<double, double>(double, double)> &&cg) : first_order_sde<RunType>(X_0, std::move(cg)) {}
    };

    template <run_type RunType>
    class first_order_sde_simulater
    {
    public:
        explicit first_order_sde_simulater(std::size_t num_steps) : num_steps_(num_steps) {}

        virtual ~first_order_sde_simulater() = default;
        virtual double simulate(const first_order_sde<RunType> &sde, double T) const = 0;

    protected:
        std::size_t num_steps_;
    };

    template <run_type RunType>
    class euler_maruyama : public first_order_sde_simulater<RunType>
    {
    public:
        euler_maruyama(std::size_t num_steps) : first_order_sde_simulater<RunType>(num_steps) {}

        double simulate(const first_order_sde<RunType> &sde, double T) const override
        {
            double dt = T / first_order_sde_simulater<RunType>::num_steps_;

            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<double> dist(0, std::sqrt(dt));

            double X_t = sde.initial_value();
            double t = 0.0;
            for (std::size_t i = 0; i < first_order_sde_simulater<RunType>::num_steps_; ++i)
            {
                auto [drift, diffusion] = sde.coefficients(X_t, t);
                double dW_t = dist(gen);
                X_t = X_t + drift * dt + diffusion * dW_t;
                t += dt;
            }
            return X_t;
        }
    };

    template <run_type RunType>
    class monte_carlo_engine final
    {
    public:
        monte_carlo_engine(std::size_t num_paths) : num_paths_(num_paths) {}

        inline double run(const std::function<double(double)> &payoff, const first_order_sde_simulater<RunType> &simulater, const first_order_sde<RunType> &sde, double T)
        {
            auto num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> ts;
            ts.reserve(num_threads);

            std::vector<std::size_t> num_sims_per_thread(num_threads, num_paths_ / num_threads);
            num_sims_per_thread[0] += num_paths_ % num_threads;

            std::vector<double> thread_res(num_threads, 0.0);
            for (std::size_t i = 0; i < num_threads; ++i)
            {
                ts.emplace_back(std::thread(
                    [&](std::size_t thread_id)
                    {
                        for (std::size_t i = 0; i < num_sims_per_thread[thread_id]; ++i)
                        {
                            thread_res[thread_id] += payoff(simulater.simulate(sde, T));
                        }
                    },
                    i));
            }
            for (auto &t : ts)
            {
                t.join();
            }

            double sum = 0.0;
            for (const auto &res : thread_res)
            {
                sum += res;
            }
            return sum /= num_paths_;
        }

    private:
        std::size_t num_paths_;
    };

} // namespace kcu
