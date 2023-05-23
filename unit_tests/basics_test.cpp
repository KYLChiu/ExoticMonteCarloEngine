#include <gtest/gtest.h>

#include <sstream>
#include <utility>

namespace {

template <typename T> class wrapper final {

public:
  explicit wrapper(T x) : x_(x) { oss_ << "Constructor\n"; }
  wrapper(wrapper &&iw) : x_(std::move(iw.x_)) { oss_ << "Move constructor\n"; }
  wrapper(const wrapper &iw) : x_(iw.x_) { oss_ << "Copy constructor\n"; }
  wrapper &operator=(wrapper iw) {
    std::swap(x_, iw.x_);
    oss_ << "Assignment\n";
    return *this;
  }
  ~wrapper() = default;

  int x() const { return x_; }
  const auto &buffer() const { return oss_; }

private:
  int x_;
  std::ostringstream oss_;
};

class implicit_string final {

public:
  implicit_string(int) { oss_ << "Implicit int constructor\n"; }
  implicit_string(const char *) { oss_ << "Implicit char* constructor\n"; }
  const auto &buffer() const { return oss_; }

private:
  std::ostringstream oss_;
};

class explicit_string final {

public:
  explicit explicit_string(int) { oss_ << "Explicit int constructor\n"; }
  explicit_string(const char *) { oss_ << "Explicit char* constructor\n"; }
  const auto &buffer() const { return oss_; }

private:
  std::ostringstream oss_;
};

} // namespace

TEST(Basics, Explicit) {
  implicit_string is = 'x';
  // explicit_string es = 'x'; does not compile
  explicit_string es = "x";
  EXPECT_EQ(is.buffer().str(), "Implicit int constructor\n");
  EXPECT_EQ(es.buffer().str(), "Explicit char* constructor\n");
}

TEST(Basics, LValueReference) {

  {
    int a;
    int &b = a;

    a = 2;
    EXPECT_EQ(b, 2);

    b = 3;
    EXPECT_EQ(a, 3);

    // int& c = 1; l-value reference must be initiaised with lvalues.
    const int &c = 1;
  }

  {
    const auto &cr = [](const int &x) { EXPECT_EQ(x, 1); };
    const auto &r = [](int &x) { EXPECT_EQ(x, 1); };

    int *p = new int(1);

    cr(1);
    cr(*p);

    // r(1); // wont compile, because non-const does not extend lifetime of
    // r-value.

    r(*p);

    delete p;
  }
}

TEST(Basics, RValueReference) {

  {
    int &&a = 1;
    EXPECT_EQ(a, 1);

    a = 2;
    EXPECT_EQ(a, 2);
  }

  {
    const auto g = []() { return 42; };
    const auto f = [](int &&a) { return a; };

    EXPECT_EQ(f(g()), 42);
    EXPECT_EQ(f(42), 42);

    int &&a = g();
    EXPECT_EQ(f(std::move(a)), 42);
  }
}

TEST(Basics, Constructors) {

  {
    wrapper iw(1);
    EXPECT_EQ(iw.buffer().str(), "Constructor\n");

    iw = std::move(iw);
    EXPECT_EQ(iw.buffer().str(), "Constructor\nAssignment\n");

    iw = iw;
    EXPECT_EQ(iw.buffer().str(), "Constructor\nAssignment\nAssignment\n");

    wrapper iw2(iw);
    EXPECT_EQ(iw2.buffer().str(), "Copy constructor\n");

    wrapper iw3(wrapper<int>{1});
    EXPECT_EQ(iw3.buffer().str(), "Constructor\n");

    wrapper iw4 = [](auto iw) { return iw; }(iw2);
    EXPECT_EQ(iw4.buffer().str(), "Move constructor\n");

    wrapper iw5 = std::move(iw);
    EXPECT_EQ(iw5.buffer().str(), "Move constructor\n");

    wrapper iw6(std::move(wrapper<int>{1}));
    EXPECT_EQ(iw6.buffer().str(), "Move constructor\n");

    wrapper iw7(std::move(1));
    EXPECT_EQ(iw7.buffer().str(), "Constructor\n");
  }
}