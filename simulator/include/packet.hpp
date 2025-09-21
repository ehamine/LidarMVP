#pragma once

#include <vector>
#include <cstdint>
#include <chrono>
#include <string>

namespace ouster_sim {

enum class PacketType {
    UNKNOWN,
    LIDAR,
    IMU
};

struct Packet {
    std::chrono::nanoseconds timestamp;
    std::string src_ip;
    std::string dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    std::vector<uint8_t> payload;
    PacketType type{PacketType::UNKNOWN};

    // Convert timestamp to seconds (double)
    double timestamp_seconds() const {
        return std::chrono::duration<double>(timestamp).count();
    }

    // Get payload size
    size_t size() const {
        return payload.size();
    }

    // Check if packet is valid
    bool is_valid() const {
        return !payload.empty() && dst_port > 0;
    }
};

} // namespace ouster_sim