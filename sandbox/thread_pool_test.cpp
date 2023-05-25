#include "kc_utils/concurrency/thread_pool.hpp"
#include <gtest/gtest.h>
#include <functional>
#include <future>
#include <ranges>

TEST(ThreadPool, ScheduleWithReturn) {
    constexpr std::size_t tp_size = 16;
    kcu::thread_pool<tp_size> tp;

    auto f = [](int i) { return i; };

    std::vector<std::future<int>> futures;
    futures.reserve(2 * tp_size);

    for (std::size_t i : std::ranges::iota_view{0UL, 2 * tp_size}) {
        futures.emplace_back(tp.schedule(f, i));
    }

    for (std::size_t i : std::ranges::iota_view{0UL, 2 * tp_size}) {
        EXPECT_EQ(futures[i].get(), i);
    }
}

TEST(ThreadPool, ScheduleWithRefReturn) {
    constexpr std::size_t tp_size = 16;
    kcu::thread_pool<tp_size> tp;

    std::atomic<int> i = 0;
    std::atomic<int> j = 0;
    auto f = [&i, &j]() -> std::atomic<int> & {
        ++i;
        return j;
    };

    std::vector<std::future<std::atomic<int> &>> futures;
    futures.reserve(2 * tp_size);
    for (std::size_t _ : std::ranges::iota_view{0UL, 2 * tp_size}) {
        futures.emplace_back(tp.schedule(f));
    }

    for (std::size_t i : std::ranges::iota_view{0UL, 2 * tp_size}) {
        EXPECT_EQ(futures[i].get(), j);
    }

    EXPECT_EQ(i, 2 * tp_size);
}

TEST(ThreadPool, ScheduleWithVoid) {
    constexpr std::size_t tp_size = 16;
    kcu::thread_pool<tp_size> tp;

    std::atomic<int> i = 0;
    auto f = [&i]() { ++i; };

    std::vector<std::future<void>> futures;
    futures.reserve(2 * tp_size);
    for (std::size_t _ : std::ranges::iota_view{0UL, 2 * tp_size}) {
        futures.emplace_back(tp.schedule(f));
    }

    for (std::size_t i : std::ranges::iota_view{0UL, 2 * tp_size}) {
        futures[i].get();
    }

    EXPECT_EQ(i, 2 * tp_size);
}
