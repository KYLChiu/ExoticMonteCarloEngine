// Copyright (c) 2023 Kelvin Yuk Lun Chiu

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:

// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <concepts>
#include <cstddef>
#include <exception>
#include <future>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace kcu {

namespace concepts {

    template <typename, template <typename...> class Template>
    struct is_template_equal : std::false_type {};

    template <template <typename...> typename Template, typename... Args>
    struct is_template_equal<Template<Args...>, Template> : std::true_type {};

    template <typename Template>
    concept future = is_template_equal<Template, std::future>::value ||
        is_template_equal<Template, std::shared_future>::value;

    template <typename F, typename T>
    concept success_callback = std::invocable<F, T>;

    template <typename F>
    concept failed_callback = std::invocable<F, std::exception_ptr>;

}  // namespace concepts

class future_chainer final {
   public:
    // Returns a future representing the evaluation of a continuation on a prior
    // (potentially shared) future.

    // Sample usage:

    // clang-format off
    // auto future = kcu::future_chainer::then(
    //     std::async([]() { return "hello"; }), [](std::string) { return 1; });
    // clang-format on

    // future.get() is expected to be 1.

    // By default, the continuation is lazily executed (i.e. the continuation
    // executes only when the future returned by this function is waited upon).
    // The continuation is executed by the calling thread of this function,
    // which may be a different thread to the one evaluating the input future.

    // If the evaluation of the input future or continuation throws an exception
    // at any point, the exception leaks into the returned future - the user is
    // expected to handle it.
    static auto then(
        concepts::future auto future,
        concepts::success_callback<decltype(future.get())> auto&& on_success,
        std::launch policy = std::launch::deferred) {
        auto continuation = [future = std::move(future),
                             on_success = std::forward<decltype(on_success)>(
                                 on_success)]() mutable {
            return on_success(future.get());
        };
        return std::async(policy, std::move(continuation));
    }

    // Returns a future representing the evaluation of a continuation on a prior
    // (potentially shared) future with exception handling. If the evaluation
    // of the input future throws an exception at any point, all
    // exceptions are caught, and a user passed in function taking in a
    // std::exception_ptr is invoked to handle the error.

    // Sample usage:

    // clang-format off
    // auto future = kcu::future_chainer::then(
    //     std::async([]() {
    //         throw std::runtime_error("Oops");
    //         return "hello";
    //     }),
    //     [](std::string) { return "Won't fire!"; }, 
    //     [](std::exception_ptr) { return "WILL fire!"; });
    // clang-format on

    // future.get() is expected to be "WILL fire!".
    static auto then(
        concepts::future auto future,
        concepts::success_callback<decltype(future.get())> auto&& on_success,
        concepts::failed_callback auto&& on_failure,
        std::launch policy = std::launch::deferred) {
        using success_t =
            std::invoke_result_t<decltype(on_success), decltype(future.get())>;
        using failed_t =
            std::invoke_result_t<decltype(on_failure), std::exception_ptr>;

        auto continuation =
            [future = std::move(future),
             on_success = std::forward<decltype(on_success)>(on_success),
             on_failure =
                 std::forward<decltype(on_failure)>(on_failure)]() mutable
            -> std::common_type_t<success_t, failed_t> {
            try {
                return on_success(future.get());
            } catch (...) {
                return on_failure(std::current_exception());
            }
        };
        return std::async(policy, std::move(continuation));
    }

    // (Sequentially) blocks for all the input futures,
    // returning the results in future of a (lazily evaluated) tuple. If the
    // return type is void, the result is returned as nullptr. The order of
    // results in the tuple is the same as the order of the variadic inputs.
    static auto gather(concepts::future auto... future) {
        auto void_handler = [](auto future) mutable {
            if constexpr (std::is_same_v<decltype(future.get()), void>) {
                future.get();
                return nullptr;
            } else {
                return future.get();
            }
        };
        auto gatherer = [... future = std::move(future),
                         vh = std::move(void_handler)]() mutable {
            return std::make_tuple(vh(future)...);
        };
        return std::async(std::launch::deferred, std::move(gatherer));
    }

    // std::vector overload of gather(), with the caveat that the futures need
    // to be of the same type (so as to be constrained in a vector).
    template <concepts::future Future>
    static auto gather(std::vector<Future> futures) {
        using T = decltype(std::declval<Future>().get());
        auto gatherer = [futures = std::move(futures)]() mutable {
            std::vector<std::decay_t<T>> values;
            values.reserve(futures.size());
            std::transform(futures.begin(), futures.end(),
                           std::back_inserter(values),
                           [](Future& future) mutable { return future.get(); });
            return values;
        };
        return std::async(std::launch::deferred, std::move(gatherer));
    }
};

}  // namespace kcu