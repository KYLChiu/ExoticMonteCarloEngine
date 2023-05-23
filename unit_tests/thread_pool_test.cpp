#include <gtest/gtest.h>

#include "kc_utils/thread_pool/thread_pool.hpp"

TEST(ThreadPool, SimpleSchedule) {
  kcu::thread_pool<1> tp;
  const auto &f = [](int i) { return i; };
  auto fut1 = tp.schedule(f, 1);
  EXPECT_EQ(fut1.get(), 1);
}
