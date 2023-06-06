#include <gtest/gtest.h>
#include "kc_utils/data_structures/linked_list.hpp"

namespace {
using int_list = kcu::linked_list<int>;
using node = int_list::node;

}  // namespace

TEST(DataStructures, LinkedListConstruction) {
    // 1 -> 2 -> 3
    auto n3 = std::make_unique<node>(3, nullptr);
    auto n2 = std::make_unique<node>(2, std::move(n3));
    auto n1 = std::make_unique<node>(1, std::move(n2));
    int_list ll(std::move(n1));

    const auto& root = ll.root();
    EXPECT_EQ(root->value(), 1);
    EXPECT_EQ(root->next()->value(), 2);
    EXPECT_EQ(root->next()->next()->value(), 3);
}

TEST(DataStructures, LinkedListInsertBeginning) {
    auto n3 = std::make_unique<node>(3, nullptr);
    auto n2 = std::make_unique<node>(2, std::move(n3));
    auto n1 = std::make_unique<node>(1, std::move(n2));
    int_list ll(std::move(n1));
    ll.insert(4, 0);

    const auto& root = ll.root();
    EXPECT_EQ(root->value(), 4);
    EXPECT_EQ(root->next()->value(), 1);
    EXPECT_EQ(root->next()->next()->value(), 2);
}

TEST(DataStructures, LinkedListInsertMiddle) {
    {
        auto n3 = std::make_unique<node>(3, nullptr);
        auto n2 = std::make_unique<node>(2, std::move(n3));
        auto n1 = std::make_unique<node>(1, std::move(n2));
        int_list ll(std::move(n1));
        ll.insert(4, 1);

        const auto& root = ll.root();
        EXPECT_EQ(root->value(), 1);
        EXPECT_EQ(root->next()->value(), 4);
        EXPECT_EQ(root->next()->next()->value(), 2);
    }

    {
        auto n3 = std::make_unique<node>(3, nullptr);
        auto n2 = std::make_unique<node>(2, std::move(n3));
        auto n1 = std::make_unique<node>(1, std::move(n2));
        int_list ll(std::move(n1));
        ll.insert(4, 2);

        const auto& root = ll.root();
        EXPECT_EQ(root->value(), 1);
        EXPECT_EQ(root->next()->value(), 2);
        EXPECT_EQ(root->next()->next()->value(), 4);
        EXPECT_EQ(root->next()->next()->next()->value(), 3);
    }
}

TEST(DataStructures, LinkedListInsertEnd) {
    auto n3 = std::make_unique<node>(3, nullptr);
    auto n2 = std::make_unique<node>(2, std::move(n3));
    auto n1 = std::make_unique<node>(1, std::move(n2));
    int_list ll(std::move(n1));
    ll.insert(4, 3);

    const auto& root = ll.root();
    EXPECT_EQ(root->value(), 1);
    EXPECT_EQ(root->next()->value(), 2);
    EXPECT_EQ(root->next()->next()->value(), 3);
    EXPECT_EQ(root->next()->next()->next()->value(), 4);
}

TEST(DataStructures, LinkedListInsertPastEnd) {
    auto n3 = std::make_unique<node>(3, nullptr);
    auto n2 = std::make_unique<node>(2, std::move(n3));
    auto n1 = std::make_unique<node>(1, std::move(n2));
    int_list ll(std::move(n1));

    EXPECT_THROW(ll.insert(4, 4), std::runtime_error);
}

TEST(DataStructures, LinkedListRemoveBeginning) {
    auto n3 = std::make_unique<node>(3, nullptr);
    auto n2 = std::make_unique<node>(2, std::move(n3));
    auto n1 = std::make_unique<node>(1, std::move(n2));
    int_list ll(std::move(n1));
    ll.remove(0);

    const auto& root = ll.root();
    EXPECT_EQ(root->value(), 2);
    EXPECT_EQ(root->next()->value(), 3);
}

TEST(DataStructures, LinkedRemoveMiddle) {
    auto n3 = std::make_unique<node>(3, nullptr);
    auto n2 = std::make_unique<node>(2, std::move(n3));
    auto n1 = std::make_unique<node>(1, std::move(n2));
    int_list ll(std::move(n1));
    ll.remove(1);

    const auto& root = ll.root();
    EXPECT_EQ(root->value(), 1);
    EXPECT_EQ(root->next()->value(), 3);
}
