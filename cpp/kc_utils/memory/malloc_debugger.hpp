#pragma once

#include <sstream>

namespace kcu {

template <typename T>
class memory_debugger {
   public:
    static std::ostringstream& oss() { return oss_; }

    void* operator new(std::size_t size) {
        oss_ << "Allocating " << size << " byte(s) with new...\n";
        return std::malloc(size);
    }

    void operator delete(void* p) noexcept {
        oss_ << "Deallocating memory with delete...\n";
        std::free(p);
    }

   private:
    static std::ostringstream oss_;
};

template <typename T>
class memory_debugger<T[]> {
   public:
    static std::ostringstream& oss() { return oss_; }

    void* operator new[](std::size_t size) {
        oss_ << "Allocating " << size << " byte(s) with new[]...\n";
        return std::malloc(size);
    }

    void operator delete[](void* p) noexcept {
        oss_ << "Deallocating memory with delete[]...\n";
        std::free(p);
    }

   private:
    static std::ostringstream oss_;
};

}  // namespace kcu
