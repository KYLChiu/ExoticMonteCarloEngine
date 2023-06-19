#pragma once

#include <memory>
#include <utility>

namespace kcu {

template <typename T>
class linked_list {
   public:
    class node final {
        friend class linked_list;

       public:
        node(T&& value, std::unique_ptr<node>&& next = nullptr)
            : value_(std::forward<T>(value)), next_(std::move(next)) {}

        auto& value() const { return value_; }
        auto& next() const { return next_; }

       private:
        T value_;
        std::unique_ptr<node> next_;
    };

    linked_list(std::unique_ptr<node>&& root) : root_(std::move(root)) {}

    auto& root() const { return root_; }

    void insert(T&& value, std::size_t pos) {
        if (pos == 0) {
            auto&& new_root = std::make_unique<node>(std::forward<T>(value),
                                                     std::move(root_));
            root_ = std::move(new_root);
            return;
        }

        auto curr = root_.get();
        std::size_t idx = 0;
        for (; curr && curr->next_.get() && idx + 1 < pos; idx++) {
            curr = curr->next_.get();
        }
        if (auto current_pos = idx + 1; idx + 1 == pos) {
            auto&& new_node = std::make_unique<node>(std::forward<T>(value),
                                                     std::move(curr->next_));
            curr->next_ = std::move(new_node);
        } else {
            throw std::runtime_error(
                "The position " + std::to_string(pos) +
                " does not exist in the list. The end is position " +
                std::to_string(current_pos) + ".");
        }
    }

    void remove(std::size_t pos) {
        if (pos == 0) {
            root_ = std::move(root_->next_);
            return;
        }

        auto prev = root_.get();
        auto curr = root_->next_.get();
        std::size_t idx = 0;
        for (; curr && curr->next_.get() && idx + 1 < pos; idx++) {
            prev = curr;
            curr = curr->next_.get();
        }
        if (auto current_pos = idx + 1; current_pos == pos) {
            prev->next_ = std::move(curr->next_);
            [[maybe_unused]] auto t =
                std::move(curr);  // make sure curr gets released.
        } else {
            throw std::runtime_error(
                "The position " + std::to_string(pos) +
                " does not exist in the list. The end is position " +
                std::to_string(current_pos) + ".");
        }
    }

   private:
    std::unique_ptr<node> root_;
};

}  // namespace kcu
