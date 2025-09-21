# Simple UDP listener for testing
add_executable(udp_listener udp_listener.cpp)
target_link_libraries(udp_listener PRIVATE Threads::Threads)

# Mock PCAP generator for testing
add_executable(mock_pcap_generator mock_pcap_generator.cpp)
target_link_libraries(mock_pcap_generator PRIVATE ouster_sim_lib)