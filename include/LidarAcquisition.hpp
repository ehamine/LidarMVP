#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include "PointTypes.hpp"

// Forward declarations
class OusterDriverEnhanced;

namespace ouster {
namespace sensor {
class client;
}
}

class LidarAcquisition {
public:
    LidarAcquisition(const std::string& hostname);
    ~LidarAcquisition();

    // Connect to the sensor
    bool connect();

    // Disconnect from the sensor
    void disconnect();

    // Check if connected
    bool isConnected() const;

    // Get sensor metadata (requires connection)
    std::string getSensorInfo() const;

    // Scan acquisition methods
    lidar_manager::PointCloudPtr getNextScan();
    bool startScanLoop(std::function<void(lidar_manager::PointCloudPtr)> callback = nullptr);
    void stopScanLoop();
    bool isScanLoopRunning() const;

    // Configuration
    void setScanTimeout(int timeout_ms);
    int getScanTimeout() const;

    // Get hostname
    const std::string& getHostname() const { return hostname_; }

    // Access to enhanced driver features
    OusterDriverEnhanced* getDriver() const { return driver_.get(); }

private:
    std::string hostname_;
    std::shared_ptr<ouster::sensor::client> client_;
    std::unique_ptr<OusterDriverEnhanced> driver_;

    // Scan acquisition state
    std::atomic<bool> scan_loop_running_;
    std::thread scan_thread_;
    int scan_timeout_ms_;
    std::function<void(lidar_manager::PointCloudPtr)> scan_callback_;

    // Private helper methods
    void scanLoopWorker();
    lidar_manager::PointCloudPtr processLidarScan(const std::string& metadata);
};