#pragma once

#include "packet.hpp"
#include <chrono>
#include <string>
#include <fstream>
#include <atomic>

namespace ouster_sim {

class MetricsLogger {
public:
    explicit MetricsLogger(const std::string& log_path = "",
                          bool verbose = false);
    ~MetricsLogger();

    // Logging methods
    void log_info(const std::string& message);
    void log_warning(const std::string& message);
    void log_error(const std::string& message);

    // Packet metrics
    void log_packet_sent(PacketType type, size_t bytes);
    void log_send_error(const std::string& error);

    // Periodic metrics (call every few seconds)
    void log_periodic_metrics();

    // Get metrics as JSON string
    std::string get_metrics_json() const;

    // Reset metrics
    void reset_metrics();

private:
    std::string log_path_;
    bool verbose_;
    std::ofstream log_file_;

    // Metrics (atomic for thread safety)
    std::atomic<uint64_t> total_packets_sent_{0};
    std::atomic<uint64_t> lidar_packets_sent_{0};
    std::atomic<uint64_t> imu_packets_sent_{0};
    std::atomic<uint64_t> total_bytes_sent_{0};
    std::atomic<uint64_t> send_errors_{0};

    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_metrics_time_;

    // Helper methods
    void write_log(const std::string& level, const std::string& message);
    std::string format_timestamp() const;
    double get_packets_per_second() const;
    double get_bytes_per_second() const;
};

} // namespace ouster_sim