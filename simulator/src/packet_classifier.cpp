#include "packet_classifier.hpp"

namespace ouster_sim {

PacketClassifier::PacketClassifier(const std::optional<SensorInfo>& sensor_info)
    : sensor_info_(sensor_info) {
}

void PacketClassifier::set_sensor_info(const SensorInfo& sensor_info) {
    sensor_info_ = sensor_info;
}

PacketType PacketClassifier::classify(const Packet& packet) const {
    stats_.total_packets++;

    PacketType type = PacketType::UNKNOWN;
    bool uncertain = false;

    // Method 1: Try classification by port (most reliable)
    type = classify_by_port(packet);
    if (type != PacketType::UNKNOWN) {
        update_stats(type, false);
        return type;
    }

    // Method 2: Try classification by sensor info (if available)
    if (sensor_info_.has_value()) {
        type = classify_by_sensor_info(packet);
        if (type != PacketType::UNKNOWN) {
            update_stats(type, false);
            return type;
        }
    }

    // Method 3: Fallback to size heuristic (less reliable)
    type = classify_by_size_heuristic(packet);
    uncertain = true;

    update_stats(type, uncertain);
    return type;
}

PacketType PacketClassifier::classify_by_port(const Packet& packet) const {
    // Standard Ouster ports
    if (packet.dst_port == 7502) {
        return PacketType::LIDAR;
    }
    if (packet.dst_port == 7503) {
        return PacketType::IMU;
    }

    return PacketType::UNKNOWN;
}

PacketType PacketClassifier::classify_by_sensor_info(const Packet& packet) const {
    if (!sensor_info_.has_value()) {
        return PacketType::UNKNOWN;
    }

    const auto& info = sensor_info_.value();

    // Check against configured ports
    if (packet.dst_port == info.udp_port_lidar) {
        return PacketType::LIDAR;
    }
    if (packet.dst_port == info.udp_port_imu) {
        return PacketType::IMU;
    }

    return PacketType::UNKNOWN;
}

PacketType PacketClassifier::classify_by_size_heuristic(const Packet& packet) const {
    size_t payload_size = packet.payload.size();

    // Heuristic based on typical Ouster packet sizes:
    // - LIDAR packets: typically 1200-1500+ bytes (depending on columns and profile)
    // - IMU packets: typically 48-100 bytes

    if (payload_size >= 1000) {
        // Large packets are likely LIDAR
        return PacketType::LIDAR;
    } else if (payload_size <= 200) {
        // Small packets are likely IMU
        return PacketType::IMU;
    }

    // Medium-sized packets are ambiguous
    return PacketType::UNKNOWN;
}

void PacketClassifier::update_stats(PacketType type, bool uncertain) const {
    switch (type) {
        case PacketType::LIDAR:
            stats_.lidar_packets++;
            break;
        case PacketType::IMU:
            stats_.imu_packets++;
            break;
        case PacketType::UNKNOWN:
            stats_.unknown_packets++;
            break;
    }

    if (uncertain) {
        stats_.uncertain_classifications++;
    }
}

} // namespace ouster_sim