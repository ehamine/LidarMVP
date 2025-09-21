#pragma once

#include <string>
#include <chrono>

namespace ouster_sim {

struct Endpoint {
    std::string ip;
    int port;

    Endpoint() : ip("127.0.0.1"), port(0) {}
    Endpoint(const std::string& ip_, int port_) : ip(ip_), port(port_) {}

    std::string to_string() const {
        return ip + ":" + std::to_string(port);
    }
};

struct Config {
    // Required
    std::string pcap_path;
    std::string json_path;

    // Network configuration
    Endpoint dst_lidar{"127.0.0.1", 7502};
    Endpoint dst_imu{"127.0.0.1", 7503};
    std::string bind_ip{"0.0.0.0"};

    // Timing
    double rate{1.0};
    bool loop{false};
    bool no_timestamps{false};
    double jitter{0.0};  // seconds
    double max_delta{1.0};  // max sleep between packets
    bool align_wallclock{false};

    // Logging
    bool verbose{false};
    std::string log_path;
    int metrics_http_port{0};  // 0 = disabled

    // Validation
    bool is_valid() const {
        return !pcap_path.empty() && rate > 0.0;
    }
};

} // namespace ouster_sim