#ifndef OUSTER_DRIVER_ENHANCED_HPP
#define OUSTER_DRIVER_ENHANCED_HPP

#include <string>
#include <memory>
#include <vector>

// Forward declarations for Ouster SDK types
namespace ouster {
namespace sensor {
struct packet_info;
struct lidar_scan;
}
}

class OusterDriverEnhanced {
public:
    OusterDriverEnhanced();
    ~OusterDriverEnhanced();

    bool init(const std::string& sensor_ip, const std::string& data_port, const std::string& imu_port);
    bool start();
    void stop();

    // Existing functionalities + new ones from spec
    // PacketLossDetector loss_detector_;
    // NetworkQualityMonitor net_monitor_;
    // AutoCalibrationManager auto_calib_;

    // New: Gestion intelligente de la connectivité
    void handleNetworkDegradation();
    void autoDetectSensorDrift();
    bool validatePacketIntegrity(const ouster::sensor::packet_info& packet);

private:
    // Ouster SDK related members
    // std::unique_ptr<ouster::sensor::client> client_;
    // std::unique_ptr<ouster::sensor::metadata> metadata_;

    // Configuration
    std::string sensor_ip_;
    std::string data_port_;
    std::string imu_port_;

    bool running_;

    // Placeholder for new modules
    // PacketLossDetector loss_detector_;
    // NetworkQualityMonitor net_monitor_;
    // AutoCalibrationManager auto_calib_;
};

#endif // OUSTER_DRIVER_ENHANCED_HPP