include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(main main.cpp price_test.cu sensitivity_test.cu)
target_link_libraries(main PRIVATE EMCE::EMCE GTest::gtest_main)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(GoogleTest)
gtest_discover_tests(main)
