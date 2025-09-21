cmake_minimum_required(VERSION 3.16)
project(ouster_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCAP REQUIRED libpcap)

find_package(Threads REQUIRED)

# Try to find nlohmann_json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    # Download nlohmann_json as fallback
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Include directories
include_directories(include)

# Source files
set(SIMULATOR_SOURCES
    src/pcap_reader.cpp
    src/sensor_info.cpp
    src/packet_classifier.cpp
    src/packet_scheduler.cpp
    src/sender_udp.cpp
    src/metrics_logger.cpp
    src/cli_parser.cpp
)

# Create library
add_library(ouster_sim_lib STATIC ${SIMULATOR_SOURCES})
target_link_libraries(ouster_sim_lib
    PRIVATE
    ${PCAP_LIBRARIES}
    nlohmann_json::nlohmann_json
    Threads::Threads
)
target_include_directories(ouster_sim_lib PRIVATE ${PCAP_INCLUDE_DIRS})
target_compile_options(ouster_sim_lib PRIVATE ${PCAP_CFLAGS_OTHER})

# Main executable
add_executable(ouster_sim src/main.cpp)
target_link_libraries(ouster_sim PRIVATE ouster_sim_lib)

# Tests
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)

    if(GTest_FOUND)
        add_subdirectory(tests)
    else()
        message(WARNING "GTest not found, skipping tests")
    endif()
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install
install(TARGETS ouster_sim DESTINATION bin)
install(FILES README.md DESTINATION share/doc/ouster_sim)