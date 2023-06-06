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
    static auto then(concepts::is_future auto future, auto&& on_success) {
        static_assert(
            std::is_invocable_v<decltype(on_success), decltype(future.get())>,
            "The success continuation is not invocable with the future return "
            "type as an argument.");

        auto continuation = [future = std::move(future),
                             on_success = std::forward<decltype(on_success)>(
                                 on_success)]() mutable {
            return on_success(future.get());
        };
        return std::async(std::move(continuation));
    }

    static auto then(concepts::is_future auto future, auto&& on_success,
                     auto&& on_failure) {
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
        return std::async(std::move(continuation));
    }

    static auto gather(concepts::is_future auto... future) {
        using C = std::common_type_t<decltype(future.get())...>;

        if constexpr (std::is_same_v<C, void>) {
            auto gatherer = [... future = std::move(future)]() mutable {
                (future.get(), ...);
            };
            return std::async(std::move(gatherer));
        } else {
            auto gatherer = [... future = std::move(future)]() mutable {
                std::vector<C> values;
                values.reserve(sizeof...(future));
                (values.emplace_back(future.get()), ...);
                return values;
            };
            return std::async(std::move(gatherer));
        }
    }
};

}  // namespace kcu