#include <gtest/gtest.h>
#include <memory>

namespace {

struct base {
    int b_ = 42;
    virtual int b() const { return b_; }
    virtual ~base() = default;
};

struct intermediate1 : virtual public base {
    int b() const override { return b_ + 1; }
};

struct intermediate2 : virtual protected base {
    int b() const override { return b_ + 2; }
};

struct intermediate3 : virtual protected base {
    int b() const override { return b_ + 3; }
};

struct derived final : public intermediate1,
                       public intermediate2,
                       public intermediate3 {
    int b() const override { return intermediate3::b() + 50; }
};

}  // namespace

TEST(Basics, MultipleInheritance) {
    {
        derived d;
        d.intermediate1::b_ = 1;
        EXPECT_EQ(d.b_, 1);
    }

    {
        auto d = std::make_shared<derived>();
        EXPECT_EQ(d->intermediate1::b(), 43);
        EXPECT_EQ(d->b(), 95);
    }
}

TEST(Basics, InheritanceCasting) {
    {
        base d = derived{};
        d.b_ = 1;
        EXPECT_EQ(d.b_, 1);
    }

    {
        std::shared_ptr<base> b = std::make_shared<derived>();
        EXPECT_EQ(b->b(), 95);

        const auto downcast = dynamic_pointer_cast<derived>(b);
        EXPECT_EQ(downcast->b(), 95);
        EXPECT_EQ(downcast->intermediate1::b(), 43);
    }

    {
        const auto& f = [](std::shared_ptr<base> p) { EXPECT_TRUE(p); };
        std::shared_ptr<base> b = std::make_shared<base>();
        const auto d = std::make_shared<derived>();
        f(b);
        f(d);
    }

    {
        const auto& f = [](base* p) { EXPECT_TRUE(p); };
        const auto b = new base;
        const auto d = new derived;
        f(b);
        f(d);
        delete b;
        delete d;
    }

    {
        const auto& f = [](const intermediate1&) {};
        const auto i1 = intermediate1{};
        const auto d = derived{};
        // f(b);
        f(i1);
        f(d);
    }
}
