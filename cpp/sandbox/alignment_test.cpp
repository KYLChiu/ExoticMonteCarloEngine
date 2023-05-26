#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <memory>
#include <new>
#include <random>
#include <ranges>
#include <thread>
#include <vector>
#include <version>

#ifdef __cpp_lib_hardware_interference_size
constexpr std::size_t hardware_constructive_interference_size =
    std::hardware_constructive_interference_size;
constexpr std::size_t hardware_destructive_interference_size =
    std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_constructive_interference_size = 64;
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

namespace {

struct align_base {
    std::size_t counter_ = 0;
};
struct align_default : public align_base {};
struct alignas(hardware_destructive_interference_size) align_cache
    : public align_base {};

template <std::size_t NumThreads, typename AlignmentType>
class alignment_tester {
   public:
    void launch() {
        const auto& do_work = [this](std::size_t thread_id) {
            std::size_t items_per_thread = num_items_ / NumThreads;
            for (std::size_t i :
                 std::ranges::iota_view{0UL, items_per_thread}) {
                partial_sums_[thread_id].counter_++;
            }
            sum_.fetch_add(partial_sums_[thread_id].counter_,
                           std::memory_order_relaxed);
        };

        std::vector<std::thread> threads;
        threads.reserve(NumThreads);
        for (std::size_t i : std::ranges::iota_view{0UL, NumThreads}) {
            threads.emplace_back(do_work, i);
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    const auto& sum() const { return sum_; }

   private:
    std::size_t num_items_ = 1 << 26;
    std::array<AlignmentType, NumThreads> partial_sums_;
    std::atomic<std::size_t> sum_;
};

TEST(FalseSharing, AlignDefault) {
    alignment_tester<16, align_default> at;
    at.launch();
    EXPECT_EQ(at.sum().load(), 67108864);
}

TEST(FalseSharing, AlignedCache) {
    alignment_tester<16, align_cache> at;
    at.launch();
    EXPECT_EQ(at.sum().load(), 67108864);
}

}  // namespace
