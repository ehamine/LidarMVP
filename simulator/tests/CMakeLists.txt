add_executable(test_ouster_sim
    test_packet_classifier.cpp
    test_sensor_info.cpp
    test_config.cpp
    test_cli_parser.cpp
)

target_link_libraries(test_ouster_sim
    PRIVATE
    ouster_sim_lib
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_ouster_sim)