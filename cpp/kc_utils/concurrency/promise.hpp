#include <any>
#include <exception>
#include <functional>
#include <future>
#include <type_traits>
#include <utility>
#include <variant>

namespace kcu {

template <typename T>
class future {
    template <typename R>
    using raw_future_t = std::future<R>;

   public:
    future() : rf_(std::make_unique<raw_future_t<T>>()) {}
    explicit future(std::shared_ptr<raw_future_t<T>>&& rf)
        : rf_(std::move(rf)) {}

    future(const future&) = delete;
    future(future&&) noexcept = default;
    future& operator=(const future&) = delete;
    future& operator=(future&&) noexcept = default;
    ~future() = default;

    template <typename F, typename... Args>
    requires std::invocable<F&&, Args&&...>
    static inline auto then(F&& f, Args&&... args) {
        using R = std::invoke_result_t<F, Args...>;
        auto g = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        auto rf = std::make_shared<raw_future_t<R>>(
            std::async([g = std::move(g)]() mutable { return g(); }));
        return future<R>(std::move(rf));
    }

    inline T get() const { return rf_->get(); }

   private:
    std::shared_ptr<raw_future_t<T>> rf_;
};

template <typename T>
class promise final {
    template <typename U>
    using data_t = std::variant<U, std::exception_ptr>;

   public:
    template <typename Resolver>
    explicit promise(Resolver resolver) {
        auto data = std::make_shared<data_t<T>>();
        auto rf = std::make_shared<std::future<std::shared_ptr<data_t<T>>>>(
            std::async([resolver, data]() mutable {
                auto fulfill = [&data](T t) mutable { *data = t; };
                auto reject = [&data](std::exception_ptr ep) mutable {
                    *data = ep;
                };
                resolver(fulfill, reject);
                return data;
            }));
        future_ =
            std::make_shared<future<std::shared_ptr<data_t<T>>>>(std::move(rf));
    }

    promise(const promise&) = delete;
    promise(promise&&) noexcept = default;
    promise& operator=(const promise&) = delete;
    promise& operator=(promise&&) noexcept = default;
    ~promise() = default;

    template <typename FulfillContinuation, typename RejectContinuation>
    requires std::invocable<FulfillContinuation&&, T&&> &&
        std::invocable<RejectContinuation&&, std::exception_ptr>
    inline auto then(FulfillContinuation&& fulfill_cont,
                     RejectContinuation&& reject_cont) {
        using F = std::invoke_result_t<FulfillContinuation&&, T&&>;
        using R =
            std::invoke_result_t<RejectContinuation&&, std::exception_ptr>;
        static_assert(std::is_same_v<F, R>,
                      "The rejected and fulfilled continuations do not return "
                      "the same type.");

        auto resolver = [fulfill_cont =
                             std::forward<FulfillContinuation>(fulfill_cont),
                         reject_cont =
                             std::forward<RejectContinuation>(reject_cont),
                         this](auto fulfill, auto reject) mutable {
            try {
                auto new_data = future_->then(
                    [](auto&& fulfill_cont, auto&& reject_cont,
                       std::shared_ptr<data_t<T>> data) mutable {
                        auto cont_res = std::make_shared<data_t<F>>();
                        try {
                            if (std::holds_alternative<T>(*data)) {
                                *cont_res = fulfill_cont(std::get<T>(*data));
                            } else {
                                *cont_res = reject_cont(
                                    std::get<std::exception_ptr>(*data));
                            }
                        } catch (...) {
                            *cont_res = reject_cont(std::current_exception());
                        }
                        return cont_res;
                    },
                    fulfill_cont, reject_cont, future_->get());
                fulfill(std::get<F>(*(new_data.get())));
            } catch (...) {
                reject(std::current_exception());
            }
        };
        return promise<F>(std::move(resolver));
    }

    inline auto get() const {
        const auto& data = future_->get();
        if (std::holds_alternative<std::exception_ptr>(*data)) {
            std::rethrow_exception(std::get<std::exception_ptr>(*data));
        } else {
            return std::get<T>(*data);
        }
    }

   private:
    std::shared_ptr<future<std::shared_ptr<data_t<T>>>> future_;
};

}  // namespace kcu