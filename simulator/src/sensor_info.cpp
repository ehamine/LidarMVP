#include "sensor_info.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>

namespace ouster_sim {

std::optional<SensorInfo> SensorInfo::from_file(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        return std::nullopt;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return from_json_string(buffer.str());
}

std::optional<SensorInfo> SensorInfo::from_json_string(const std::string& json_str) {
    try {
        auto json_data = nlohmann::json::parse(json_str);
        SensorInfo info;

        // Required fields
        if (json_data.contains("prod_sn")) {
            info.sensor_name = json_data["prod_sn"].get<std::string>();
        } else if (json_data.contains("sensor_name")) {
            info.sensor_name = json_data["sensor_name"].get<std::string>();
        }

        // Lidar mode
        if (json_data.contains("lidar_mode")) {
            info.lidar_mode = json_data["lidar_mode"].get<std::string>();
        }

        // UDP ports
        if (json_data.contains("udp_port_lidar")) {
            info.udp_port_lidar = json_data["udp_port_lidar"].get<int>();
        }
        if (json_data.contains("udp_port_imu")) {
            info.udp_port_imu = json_data["udp_port_imu"].get<int>();
        }

        // Beam information
        if (json_data.contains("beam_azimuth_angles")) {
            info.beam_azimuth = json_data["beam_azimuth_angles"].get<std::vector<double>>();
        }
        if (json_data.contains("beam_altitude_angles")) {
            info.beam_altitude = json_data["beam_altitude_angles"].get<std::vector<double>>();
        }

        // Calculate number of lasers
        if (!info.beam_altitude.empty()) {
            info.num_lasers = static_cast<int>(info.beam_altitude.size());
        } else if (!info.beam_azimuth.empty()) {
            info.num_lasers = static_cast<int>(info.beam_azimuth.size());
        } else if (json_data.contains("beam_to_lidar_transform")) {
            // Count the transform matrices (each laser has one)
            auto transforms = json_data["beam_to_lidar_transform"];
            if (transforms.is_array()) {
                info.num_lasers = static_cast<int>(transforms.size() / 16); // 4x4 matrix = 16 elements
            }
        }

        // Fallback: try to infer from lidar_mode
        if (info.num_lasers == 0 && !info.lidar_mode.empty()) {
            if (info.lidar_mode.find("OS1-32") != std::string::npos) {
                info.num_lasers = 32;
            } else if (info.lidar_mode.find("OS1-64") != std::string::npos) {
                info.num_lasers = 64;
            } else if (info.lidar_mode.find("OS1-128") != std::string::npos) {
                info.num_lasers = 128;
            } else if (info.lidar_mode.find("OS2-32") != std::string::npos) {
                info.num_lasers = 32;
            } else if (info.lidar_mode.find("OS2-64") != std::string::npos) {
                info.num_lasers = 64;
            } else if (info.lidar_mode.find("OS2-128") != std::string::npos) {
                info.num_lasers = 128;
            }
        }

        return info;

    } catch (const std::exception&) {
        return std::nullopt;
    }
}

int SensorInfo::get_expected_packets_per_scan() const {
    // Heuristic based on common Ouster configurations
    if (lidar_mode.find("512") != std::string::npos) {
        return 512; // 512 columns per revolution
    } else if (lidar_mode.find("1024") != std::string::npos) {
        return 1024; // 1024 columns per revolution
    } else if (lidar_mode.find("2048") != std::string::npos) {
        return 2048; // 2048 columns per revolution
    }

    // Default estimate
    return 512;
}

std::string SensorInfo::to_string() const {
    std::stringstream ss;
    ss << "SensorInfo{\n";
    ss << "  sensor_name: " << sensor_name << "\n";
    ss << "  lidar_mode: " << lidar_mode << "\n";
    ss << "  num_lasers: " << num_lasers << "\n";
    ss << "  udp_port_lidar: " << udp_port_lidar << "\n";
    ss << "  udp_port_imu: " << udp_port_imu << "\n";
    ss << "  beam_azimuth.size(): " << beam_azimuth.size() << "\n";
    ss << "  beam_altitude.size(): " << beam_altitude.size() << "\n";
    ss << "  expected_packets_per_scan: " << get_expected_packets_per_scan() << "\n";
    ss << "}";
    return ss.str();
}

} // namespace ouster_sim