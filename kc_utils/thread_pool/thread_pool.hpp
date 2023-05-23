#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <ranges>
#include <semaphore>
#include <thread>

namespace kcu {

template <unsigned N>
class thread_pool final {
   public:
    explicit thread_pool() : active_(true) {
        for (std::size_t i : std::ranges::iota_view{0UL, N}) {
            threads_[i] = std::thread(&thread_pool::worker_thread, this);
        }
    }

    // No copy/move constructors.
    thread_pool(const thread_pool &) = delete;
    thread_pool &operator=(const thread_pool &) = delete;

    ~thread_pool() {
        active_ = false;
        cs_.release(N);
        for (auto &t : threads_) {
            t.join();
        }
    }

    template <typename F, typename... Args>
    requires std::invocable<F, Args...>
    auto schedule(F &&f, Args &&...args) {
        using R = std::invoke_result_t<F, Args...>;
        auto bound_f =
            std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        auto task =
            std::make_shared<std::packaged_task<R()>>(std::move(bound_f));
        auto future = task->get_future();
        task_queue_.push([task = std::move(task)]() { task->operator()(); });
        cs_.release();
        return future;
    }

   private:
    void worker_thread() {
        std::function<void()> task;
        while (active_) {
            cs_.acquire();
            if (active_) {
                {
                    std::unique_lock ul(mtx_);
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
                task();
            }
        }
    }

    std::atomic<bool> active_;
    std::queue<std::function<void()>> task_queue_;
    std::array<std::thread, N> threads_;
    std::counting_semaphore<N> cs_{0};
    std::mutex mtx_;
};

}  // namespace kcu