#include "LidarAcquisition.hpp"
#include "OusterDriverEnhanced.hpp"
#include <ouster/client.h>
#include <ouster/types.h>
#include <ouster/lidar_scan.h>
#include <iostream>
#include <fmt/core.h>
#include <thread>
#include <chrono>
#include <cmath>

#ifndef UDP_BUF_SIZE
#define UDP_BUF_SIZE 65536
#endif

LidarAcquisition::LidarAcquisition(const std::string& hostname)
    : hostname_(hostname), client_(nullptr), driver_(std::make_unique<OusterDriverEnhanced>()),
      scan_loop_running_(false), scan_timeout_ms_(1000) {}

bool LidarAcquisition::connect() {
    if (hostname_.empty()) {
        fmt::print(stderr, "Error: Empty hostname provided\n");
        return false;
    }

    fmt::print("Connecting to Lidar sensor at {}...\n", hostname_);

    try {
        // Initialize enhanced driver first
        if (!driver_->init(hostname_, "7502", "7503")) {
            fmt::print(stderr, "Failed to initialize enhanced driver for {}\n", hostname_);
            return false;
        }

        // Initialize Ouster client with default ports
        client_ = ouster::sensor::init_client(hostname_, 7502, 7503);
        if (!client_) {
            fmt::print(stderr, "Failed to initialize Ouster client for {}\n", hostname_);
            return false;
        }

        // Start enhanced driver
        if (!driver_->start()) {
            fmt::print(stderr, "Failed to start enhanced driver\n");
            client_.reset();
            return false;
        }

        fmt::print("Successfully connected to sensor at {} with enhanced features\n", hostname_);
        return true;
    }
    catch (const std::exception& e) {
        fmt::print(stderr, "Exception during connection: {}\n", e.what());
        return false;
    }
}

void LidarAcquisition::disconnect() {
    if (client_) {
        fmt::print("Disconnecting from sensor...\n");

        // Stop scan loop if running
        stopScanLoop();

        // Stop enhanced driver first
        if (driver_) {
            driver_->stop();
        }

        // Then cleanup client
        client_.reset();
        fmt::print("Disconnected from {}\n", hostname_);
    }
}

bool LidarAcquisition::isConnected() const {
    return client_ != nullptr;
}

std::string LidarAcquisition::getSensorInfo() const {
    if (!client_) {
        return "Not connected to sensor";
    }

    try {
        // Get metadata from sensor
        auto metadata_str = ouster::sensor::get_metadata(*client_);

        // For now, return basic info with the metadata string length
        // In a real implementation, we'd parse the JSON metadata
        std::string info = fmt::format(
            "Sensor: {}\n"
            "Status: Connected\n"
            "Metadata size: {} bytes\n",
            hostname_,
            metadata_str.length()
        );

        return info;
    }
    catch (const std::exception& e) {
        return fmt::format("Error getting sensor info: {}", e.what());
    }
}

// Scan acquisition methods
lidar_manager::PointCloudPtr LidarAcquisition::getNextScan() {
    if (!client_) {
        fmt::print(stderr, "Error: Not connected to sensor\n");
        return nullptr;
    }

    try {
        // Get metadata for configuration info
        auto metadata_str = ouster::sensor::get_metadata(*client_);

        // For now, generate sample data instead of real acquisition
        // This is a placeholder implementation that will be replaced with real SDK calls
        auto point_cloud = processLidarScan(metadata_str);

        fmt::print("Acquired scan with {} points\n", point_cloud ? point_cloud->size() : 0);
        return point_cloud;

    } catch (const std::exception& e) {
        fmt::print(stderr, "Error getting scan: {}\n", e.what());
        return nullptr;
    }
}

bool LidarAcquisition::startScanLoop(std::function<void(lidar_manager::PointCloudPtr)> callback) {
    if (!client_) {
        fmt::print(stderr, "Error: Not connected to sensor\n");
        return false;
    }

    if (scan_loop_running_) {
        fmt::print("Scan loop is already running\n");
        return true;
    }

    scan_callback_ = callback;
    scan_loop_running_ = true;

    // Start scan thread
    scan_thread_ = std::thread(&LidarAcquisition::scanLoopWorker, this);

    fmt::print("Scan loop started\n");
    return true;
}

void LidarAcquisition::stopScanLoop() {
    if (!scan_loop_running_) {
        return;
    }

    fmt::print("Stopping scan loop...\n");
    scan_loop_running_ = false;

    if (scan_thread_.joinable()) {
        scan_thread_.join();
    }

    fmt::print("Scan loop stopped\n");
}

bool LidarAcquisition::isScanLoopRunning() const {
    return scan_loop_running_;
}

void LidarAcquisition::setScanTimeout(int timeout_ms) {
    scan_timeout_ms_ = timeout_ms;
}

int LidarAcquisition::getScanTimeout() const {
    return scan_timeout_ms_;
}

// Private helper methods
void LidarAcquisition::scanLoopWorker() {
    fmt::print("Scan loop worker started\n");

    while (scan_loop_running_) {
        auto scan = getNextScan();
        if (scan && scan_callback_) {
            scan_callback_(scan);
        }

        // Small delay to prevent excessive CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    fmt::print("Scan loop worker finished\n");
}

lidar_manager::PointCloudPtr LidarAcquisition::processLidarScan(const std::string& metadata) {
    auto point_cloud = std::make_shared<lidar_manager::PointCloud>();

    try {
        // Generate sample points for demonstration
        // In a real implementation, we would parse the actual Ouster scan data
        point_cloud->clear();

        // Default configuration for demonstration (512x64 typical Ouster config)
        int columns_per_frame = 512;
        int pixels_per_column = 64;
        point_cloud->reserve(columns_per_frame * pixels_per_column);

        // Create sample scan pattern
        uint32_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        for (int ring = 0; ring < pixels_per_column; ++ring) {
            for (int col = 0; col < columns_per_frame; ++col) {
                // Generate sample 3D coordinates
                float angle_rad = (col * 2.0f * M_PI) / columns_per_frame;
                float range = 10.0f + (ring * 0.5f); // Sample range

                float x = range * cos(angle_rad);
                float y = range * sin(angle_rad);
                float z = ring * 0.1f - 3.2f; // Sample elevation

                uint16_t intensity = 1000 + (ring * 10) + (col % 100);

                point_cloud->addPoint(x, y, z, intensity, ring, timestamp + col);
            }
        }

        // Update metadata
        auto& metadata_ref = point_cloud->getMetadata();
        metadata_ref.scan_id = timestamp;
        metadata_ref.points_count = point_cloud->size();
        metadata_ref.lidar_mode = 1; // Default mode

        fmt::print("Generated sample point cloud with {} points\n", point_cloud->size());

    } catch (const std::exception& e) {
        fmt::print(stderr, "Error processing scan: {}\n", e.what());
        return nullptr;
    }

    return point_cloud;
}

LidarAcquisition::~LidarAcquisition() {
    disconnect();
}
