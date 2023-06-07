#pragma once

#include <concepts>
#include <exception>
#include <future>
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
    concept is_future = is_template_equal<Template, std::future>::value ||
        is_template_equal<Template, std::shared_future>::value;

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
    static auto then(concepts::is_future auto future, auto&& on_success,
                     std::launch policy = std::launch::deferred) {
        static_assert(
            std::is_invocable_v<decltype(on_success), decltype(future.get())>,
            "The success continuation is not invocable with the future return "
            "type as an argument.");

        auto continuation = [future = std::move(future),
                             on_success = std::forward<decltype(on_success)>(
                                 on_success)]() mutable {
            return on_success(future.get());
        };
        return std::async(policy, std::move(continuation));
    }

    // Returns a future representing the evaluation of a continuation on a prior
    // (potentially shared) future with exception handling. If the evaluation
    // of the input future or continuation throws an exception at any point, all
    // exceptions are caught, and a user passed in function taking in a
    // std::exception_ptr is invoked to handle the error.

    // Sample usage:

    // clang-format off
    // auto future = kcu::future_chainer::then(
    //     std::async([]() {
    //         throw std::runtime_error("Oops");
    //         return "hello";
    //     }),
    //     [](std::string) { return "Won't fire!"; }, [](std::exception_ptr) { return 2; });
    // clang-format on

    // future.get() is expected to be 2.
    static auto then(concepts::is_future auto future, auto&& on_success,
                     auto&& on_failure,
                     std::launch policy = std::launch::deferred) {
        using T = decltype(future.get());

        static_assert(
            std::is_invocable_v<decltype(on_success), T>,
            "The success continuation is not invocable with the future return "
            "type as an argument.");
        static_assert(
            std::is_invocable_v<decltype(on_failure), std::exception_ptr>,
            "The failure continuation is not invocable with an exception "
            "pointer as an argument.");

        using S = std::invoke_result_t<decltype(on_success), T>;
        using F =
            std::invoke_result_t<decltype(on_failure), std::exception_ptr>;

        auto continuation =
            [future = std::move(future),
             on_success = std::forward<decltype(on_success)>(on_success),
             on_failure = std::forward<decltype(on_failure)>(
                 on_failure)]() mutable -> std::common_type_t<S, F> {
            try {
                return on_success(future.get());
            } catch (...) {
                return on_failure(std::current_exception());
            }
        };
        return std::async(policy, std::move(continuation));
    }

    // (Sequentially) blocks for all the input futures,
    // returning the results in a vector. The order of results in the vector is
    // the same as the order of the variadic inputs. Lazyily evaluated function.
    static auto gather(concepts::is_future auto... future) {
        using T = std::common_type_t<decltype(future.get())...>;

        if constexpr (std::is_same_v<T, void>) {
            auto gatherer = [... future = std::move(future)]() mutable {
                (future.get(), ...);
            };
            return std::async(std::launch::deferred, std::move(gatherer));
        } else {
            auto gatherer = [... future = std::move(future)]() mutable {
                std::vector<T> values;
                values.reserve(sizeof...(future));
                (values.emplace_back(future.get()), ...);
                return values;
            };
            return std::async(std::launch::deferred, std::move(gatherer));
        }
    }

    // Vector overload of gather().
    template <concepts::is_future Future>
    static auto gather(std::vector<Future> futures) {
        using T = decltype(std::declval<typename std::decay<
                               decltype(*futures.begin())>::type>()
                               .get());

        if constexpr (std::is_same_v<T, void>) {
            auto gatherer = [futures = std::move(futures)]() mutable {
                for (auto& f : futures) {
                    f.get();
                }
            };
            return std::async(std::launch::deferred, std::move(gatherer));
        } else {
            auto gatherer = [futures = std::move(futures)]() mutable {
                std::vector<std::decay_t<T>> values;
                values.reserve(futures.size());
                std::transform(
                    futures.begin(), futures.end(), std::back_inserter(values),
                    [](auto& future) mutable { return future.get(); });
                return values;
            };
            return std::async(std::launch::deferred, std::move(gatherer));
        }
    }
};

}  // namespace kcu