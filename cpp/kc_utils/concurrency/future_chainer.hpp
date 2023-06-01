#include <concepts>
#include <exception>
#include <future>
#include <type_traits>
#include <utility>

namespace kcu {

template <typename U, typename... V>
concept is_any_of = (std::same_as<U, V> || ...);

template <template <typename> typename Future, typename T>
concept is_future = is_any_of<Future<T>, std::future<T>, std::shared_future<T>>;

class future_chainer final {
   public:
    template <template <typename> typename Future, typename T>
    requires is_future<Future, T>
    static auto then(Future<T> future, std::invocable<T> auto&& on_success) {
        auto continuation = [future = std::move(future),
                             on_success = std::forward<decltype(on_success)>(
                                 on_success)]() mutable {
            return on_success(future.get());
        };
        return std::async(std::move(continuation));
    }

    template <template <typename> typename Future, typename T>
    requires is_future<Future, T>
    static auto then(Future<T> future, std::invocable<T> auto&& on_success,
                     std::invocable<std::exception_ptr> auto&& on_failure) {
        using RetSuccess = std::invoke_result_t<decltype(on_success), T>;
        using RetFailure =
            std::invoke_result_t<decltype(on_failure), std::exception_ptr>;
        static_assert(std::is_convertible_v<RetSuccess, RetFailure>,
                      "The common type of the success and failure "
                      "continuations is not well defined.");
        using Ret = std::common_type_t<RetSuccess, RetFailure>;

        auto continuation = [future = std::move(future),
                             on_success =
                                 std::forward<decltype(on_success)>(on_success),
                             on_failure = std::forward<decltype(on_failure)>(
                                 on_failure)]() mutable -> Ret {
            try {
                return on_success(future.get());
            } catch (...) {
                return on_failure(std::current_exception());
            }
        };
        return std::async(std::move(continuation));
    }
};

}  // namespace kcu