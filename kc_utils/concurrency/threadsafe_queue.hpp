#pragma once

#include <mutex>
#include <queue>
#include <semaphore>

namespace kcu {

template <typename T>
class threadsafe_queue final {
   public:
    threadsafe_queue() = default;

    void push(T&& t) {
        {
            std::lock_guard lock(mtx_);
            queue_.push(std::forward<T>(t));
        }
        bs_.release();
    }

    const T& front() const {
        std::lock_guard lock(mtx_);
        bs_.acquire();
        return queue_.front();
    }

    void pop() {
        std::lock_guard lock(mtx_);
        queue_.pop();
    }

   private:
    mutable std::mutex mtx_;
    mutable std::binary_semaphore bs_{0};
    std::queue<T> queue_;
};

}  // namespace kcu