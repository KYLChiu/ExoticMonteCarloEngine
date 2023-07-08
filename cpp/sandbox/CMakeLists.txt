include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
  main
  main.cpp
  mc_pricer_test.cu
)

target_link_libraries(
  main
  GTest::gtest_main
  ${CUDA_LIBRARIES}
)
target_include_directories(main PUBLIC "${PROJECT_SOURCE_DIR}")

include(GoogleTest)
gtest_discover_tests(main)


