#pragma once

#include <string>
#include <vector>
#include <optional>

namespace ouster_sim {

struct SensorInfo {
    std::string sensor_name;
    std::string lidar_mode;
    int udp_port_lidar{7502};
    int udp_port_imu{7503};
    std::vector<double> beam_azimuth;
    std::vector<double> beam_altitude;
    int num_lasers{0};

    // Static factory method to create from JSON file
    static std::optional<SensorInfo> from_file(const std::string& json_path);

    // Static factory method to create from JSON string
    static std::optional<SensorInfo> from_json_string(const std::string& json_str);

    // Validation
    bool is_valid() const {
        return !sensor_name.empty() && num_lasers > 0;
    }

    // Get expected packets per scan (heuristic)
    int get_expected_packets_per_scan() const;

    // Debug info
    std::string to_string() const;
};

} // namespace ouster_sim