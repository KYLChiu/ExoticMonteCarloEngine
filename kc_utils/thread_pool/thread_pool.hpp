#pragma once

#include <array>
#include <concepts>
#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <ranges>
#include <semaphore>
#include <thread>

namespace kcu {

template <unsigned N> class thread_pool final {

public:
  explicit thread_pool() {
    std::size_t num_threads = N > std::thread::hardware_concurrency()
                                  ? std::thread::hardware_concurrency()
                                  : N;
    for (std::size_t i :
         std::ranges::iota_view{static_cast<std::size_t>(0), num_threads}) {
      threads_[i] = std::thread(&thread_pool::worker_thread, this);
    }
  }

  // No copy/move constructors.
  thread_pool(const thread_pool &) = delete;
  thread_pool &operator=(const thread_pool &) = delete;

  ~thread_pool() {
    for (auto &t : threads_) {
      t.detach();
    }
  }

  template <typename F, typename... Args>
  requires std::invocable<F, Args...>
  auto schedule(F &&f, Args &&...args) {
    using R = std::invoke_result_t<F, Args...>;
    auto pm = std::make_shared<std::promise<R>>();
    auto bound_f = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    auto task = [pm, bound_f = std::move(bound_f)]() mutable {
      try {
        std::cout << "EXECUTING" << std::endl;
        pm->set_value(bound_f());
      } catch (...) {
        try {
          pm->set_exception(std::current_exception());
        } catch (...) {
          // TODO: what to do if this throws?
        }
      }
    };
    cs_.release();
    task_queue_.push(std::move(task));
    return pm->get_future();
  }

private:
  void worker_thread() {
    while (true) {
      cs_.acquire();
      auto task = std::move(task_queue_.front());
      task_queue_.pop();
      task();
    }
  }

  std::queue<std::function<void()>> task_queue_;
  std::array<std::thread, N> threads_;
  std::counting_semaphore<N> cs_{0};
};

} // namespace kcu