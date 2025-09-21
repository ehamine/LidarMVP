add_executable(run_tests test_lidar_acquisition.cpp)
target_link_libraries(run_tests PRIVATE GTest::gtest GTest::gtest_main lidar_manager_core)
include(GoogleTest)
gtest_discover_tests(run_tests)