#pragma once

#include "packet.hpp"
#include "sensor_info.hpp"
#include <optional>

namespace ouster_sim {

class PacketClassifier {
public:
    explicit PacketClassifier(const std::optional<SensorInfo>& sensor_info = std::nullopt);

    // Classify packet as LIDAR or IMU
    PacketType classify(const Packet& packet) const;

    // Set sensor info (can be called later)
    void set_sensor_info(const SensorInfo& sensor_info);

    // Statistics
    struct Stats {
        uint64_t total_packets{0};
        uint64_t lidar_packets{0};
        uint64_t imu_packets{0};
        uint64_t unknown_packets{0};
        uint64_t uncertain_classifications{0};
    };

    const Stats& get_stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    std::optional<SensorInfo> sensor_info_;
    mutable Stats stats_;

    // Classification methods
    PacketType classify_by_port(const Packet& packet) const;
    PacketType classify_by_size_heuristic(const Packet& packet) const;
    PacketType classify_by_sensor_info(const Packet& packet) const;

    void update_stats(PacketType type, bool uncertain = false) const;
};

} // namespace ouster_sim