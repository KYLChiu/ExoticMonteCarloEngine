#include "kc_utils/memory/unique_ptr.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {

struct X {
    X(int x) : x_(x) {}
    int x() const { return x_; }

    bool operator==(const X& x) { return x_ == x.x_; }

   private:
    int x_;
};

}  // namespace

TEST(SmartPointer, UniquePtr) {
    // make_unique
    {
        auto u = kcu::make_unique<int>(1);
        EXPECT_EQ(*u, 1);
    }

    // get
    {
        auto u = kcu::make_unique<int>(1);
        auto p = u.get();
        EXPECT_EQ(*p, 1);
    }

    // Move ctor
    { kcu::unique_ptr<int> u(kcu::make_unique<int>(1)); }

    // Move assignment
    {
        auto u1 = kcu::make_unique<int>(1);
        kcu::unique_ptr<int> u2 = nullptr;
        u2 = std::move(u1);
        EXPECT_FALSE(u1);
    }

    // Custom deleter
    {
        std::ostringstream oss;

        const auto& del = [&oss](auto* ptr) {
            oss << "Custom deleter...";
            delete ptr;
        };

        {
            int* p = new int(1);
            kcu::unique_ptr<int, decltype(del)>(p, del);
        }

        EXPECT_EQ(oss.str(), "Custom deleter...");
    }

    // Operators
    {
        auto u1 = kcu::make_unique<X>(1);
        auto u2 = kcu::make_unique<X>(1);
        EXPECT_TRUE(u1 == u1);
        EXPECT_TRUE(u1 != u2);
        EXPECT_EQ(u1->x(), 1);
        EXPECT_EQ((*u1).x(), 1);
    }
}
