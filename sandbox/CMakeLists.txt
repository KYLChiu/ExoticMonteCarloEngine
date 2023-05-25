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
  unique_ptr_test.cpp
  basics_test.cpp
  inheritance_test.cpp
  thread_pool_test.cpp
)
target_link_libraries(
  main
  GTest::gtest_main
)

message(${PROJECT_SOURCE_DIR})
target_include_directories(main PUBLIC
        "${PROJECT_SOURCE_DIR}"
        )

include(GoogleTest)
gtest_discover_tests(main)
