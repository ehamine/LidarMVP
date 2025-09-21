#include "metrics_logger.hpp"
#include <iomanip>
#include <sstream>
#include <iostream>

namespace ouster_sim {

MetricsLogger::MetricsLogger(const std::string& log_path, bool verbose)
    : log_path_(log_path), verbose_(verbose),
      start_time_(std::chrono::steady_clock::now()),
      last_metrics_time_(start_time_) {

    if (!log_path_.empty()) {
        log_file_.open(log_path_, std::ios::app);
        if (log_file_.is_open()) {
            log_info("Metrics logger started");
        }
    }
}

MetricsLogger::~MetricsLogger() {
    if (log_file_.is_open()) {
        log_info("Metrics logger stopped");
        log_file_.close();
    }
}

void MetricsLogger::log_info(const std::string& message) {
    write_log("INFO", message);
}

void MetricsLogger::log_warning(const std::string& message) {
    write_log("WARN", message);
}

void MetricsLogger::log_error(const std::string& message) {
    write_log("ERROR", message);
}

void MetricsLogger::log_packet_sent(PacketType type, size_t bytes) {
    total_packets_sent_++;
    total_bytes_sent_ += bytes;

    switch (type) {
        case PacketType::LIDAR:
            lidar_packets_sent_++;
            break;
        case PacketType::IMU:
            imu_packets_sent_++;
            break;
        case PacketType::UNKNOWN:
            break;
    }
}

void MetricsLogger::log_send_error(const std::string& error) {
    send_errors_++;
    log_error("Send error: " + error);
}

void MetricsLogger::log_periodic_metrics() {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = current_time - last_metrics_time_;

    if (elapsed >= std::chrono::seconds(5)) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "METRICS: ";
        ss << "packets_sent=" << total_packets_sent_.load();
        ss << ", lidar=" << lidar_packets_sent_.load();
        ss << ", imu=" << imu_packets_sent_.load();
        ss << ", rate=" << get_packets_per_second() << " pps";
        ss << ", throughput=" << (get_bytes_per_second() / 1024.0 / 1024.0) << " MB/s";
        ss << ", errors=" << send_errors_.load();

        log_info(ss.str());
        last_metrics_time_ = current_time;
    }
}

std::string MetricsLogger::get_metrics_json() const {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"total_packets_sent\": " << total_packets_sent_.load() << ",\n";
    ss << "  \"lidar_packets_sent\": " << lidar_packets_sent_.load() << ",\n";
    ss << "  \"imu_packets_sent\": " << imu_packets_sent_.load() << ",\n";
    ss << "  \"total_bytes_sent\": " << total_bytes_sent_.load() << ",\n";
    ss << "  \"send_errors\": " << send_errors_.load() << ",\n";
    ss << "  \"packets_per_second\": " << get_packets_per_second() << ",\n";
    ss << "  \"bytes_per_second\": " << get_bytes_per_second() << ",\n";

    auto now = std::chrono::steady_clock::now();
    auto runtime = std::chrono::duration<double>(now - start_time_).count();
    ss << "  \"runtime_seconds\": " << runtime << "\n";
    ss << "}";

    return ss.str();
}

void MetricsLogger::reset_metrics() {
    total_packets_sent_ = 0;
    lidar_packets_sent_ = 0;
    imu_packets_sent_ = 0;
    total_bytes_sent_ = 0;
    send_errors_ = 0;
    start_time_ = std::chrono::steady_clock::now();
    last_metrics_time_ = start_time_;
}

void MetricsLogger::write_log(const std::string& level, const std::string& message) {
    std::string log_line = format_timestamp() + " [" + level + "] " + message;

    if (verbose_) {
        std::cout << log_line << std::endl;
    }

    if (log_file_.is_open()) {
        log_file_ << log_line << std::endl;
        log_file_.flush();
    }
}

std::string MetricsLogger::format_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();

    return ss.str();
}

double MetricsLogger::get_packets_per_second() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(now - start_time_).count();
    return duration > 0.0 ? static_cast<double>(total_packets_sent_.load()) / duration : 0.0;
}

double MetricsLogger::get_bytes_per_second() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(now - start_time_).count();
    return duration > 0.0 ? static_cast<double>(total_bytes_sent_.load()) / duration : 0.0;
}

} // namespace ouster_sim