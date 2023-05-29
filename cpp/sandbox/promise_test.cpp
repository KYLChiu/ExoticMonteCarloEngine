#include "kc_utils/concurrency/promise.hpp"
#include <gtest/gtest.h>
#include <ranges>

TEST(Future, Value) {
    kcu::future<int> future;
    auto next_future = future.then([](auto i) { return i; }, 5);
    EXPECT_EQ(next_future.get(), 5);
}

TEST(Future, VoidRef) {
    kcu::future<void> fut;
    int j = 0;
    for ([[maybe_unused]] int i : std::ranges::iota_view{1, 10}) {
        fut.then([&j]() mutable { ++j; });
    }
    fut.then([&j]() mutable { ++j; }).get();
    EXPECT_EQ(j, 10);
}

TEST(Future, ChangeType) {
    kcu::future<int> future;
    auto next_future =
        future.then([](auto i, auto j) { return std::make_pair(i, j); }, 5, 5);
    auto pair = next_future.get();
    EXPECT_EQ(pair, std::make_pair(5, 5));
}

TEST(Promise, Fulfill) {
    int value = 5;
    kcu::promise<int> promise([=](auto fulfill, auto) { fulfill(value); });
    auto v = promise.get();
    EXPECT_EQ(v, value);
}

TEST(Promise, Reject) {
    const char* const exception_msg = "Oh no!";
    auto f = [=](auto, auto reject) {
        reject(std::make_exception_ptr(std::runtime_error(exception_msg)));
    };
    kcu::promise<int> promise(f);
    EXPECT_THROW(promise.get(), std::runtime_error);
}

TEST(Promise, FulfillContinuation) {
    int value = 12;
    kcu::promise<int> promise(
        [=](auto fulfill, auto) { return fulfill(value); });

    auto next_promise =
        promise.then([](int i) { return i + 1; },
                     [](std::exception_ptr) { return INT_MAX; });

    EXPECT_EQ(next_promise.get(), 13);
}

TEST(Promise, RejectContinuation) {
    const char* const exception_msg = "Oh no!";
    auto f = [=](auto, auto reject) {
        reject(std::make_exception_ptr(std::runtime_error(exception_msg)));
    };
    kcu::promise<int> promise(f);

    auto next_promise =
        promise.then([](int i) { return i + 1; },
                     [](std::exception_ptr) { return INT_MAX; });

    EXPECT_EQ(next_promise.get(), INT_MAX);
}

TEST(Promise, ChangeType) {
    kcu::promise<int> promise([](auto fulfill, auto) { return fulfill(0); });
    auto next_promise =
        promise.then([](int) { return "Hello!"; },
                     [](std::exception_ptr) { return "Should not fire."; });

    EXPECT_EQ(next_promise.get(), "Hello!");
}

// TEST(Promise, Chain) {
//     auto promise =
//         kcu::promise<int>([](auto fulfill, auto reject) {
//             std::cout << "RESOLVER" << std::endl;
//             return fulfill(0);
//         })
//             .then(
//                 [](int i) {
//                     std::cout << "CONT" << std::endl;
//                     return std::string{"CONT"};
//                 },
//                 [](std::exception_ptr ep) { return std::string{""}; });
//     EXPECT_EQ(promise.get(), "CONT");
// }