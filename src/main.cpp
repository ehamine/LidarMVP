#include <iostream>
#include <fmt/core.h>
#include <thread>
#include <chrono>
#include "LidarAcquisition.hpp"
#include "OusterDriverEnhanced.hpp"

int main(int argc, char *argv[]) {
    std::string version = "2.0";
    fmt::print("Hello Lidar Manager V{}!\n", version);

    if (argc < 2) {
        fmt::print("Usage: {} <sensor_hostname>\n", argv[0]);
        return 1;
    }

    std::string sensor_hostname = argv[1];
    LidarAcquisition lidar_acquisition(sensor_hostname);

    if (lidar_acquisition.connect()) {
        fmt::print("Connection to sensor successful.\n");

        // Test enhanced driver features
        auto* driver = lidar_acquisition.getDriver();
        if (driver) {
            fmt::print("Testing enhanced driver features...\n");
            driver->handleNetworkDegradation();
            driver->autoDetectSensorDrift();
        }

        // Get sensor information
        fmt::print("Sensor Information:\n{}\n", lidar_acquisition.getSensorInfo());

        // Test single scan acquisition
        fmt::print("Testing single scan acquisition...\n");
        auto scan = lidar_acquisition.getNextScan();
        if (scan) {
            fmt::print("Successfully acquired scan with {} points\n", scan->size());
            fmt::print("Average intensity: {:.2f}\n", scan->computeAverageIntensity());

            // Show bounding box
            lidar_manager::PointXYZIRT min_point, max_point;
            scan->computeBoundingBox(min_point, max_point);
            fmt::print("Bounding box: [{:.2f}, {:.2f}, {:.2f}] to [{:.2f}, {:.2f}, {:.2f}]\n",
                      min_point.x, min_point.y, min_point.z,
                      max_point.x, max_point.y, max_point.z);
        }

        // Test scan loop for a few seconds
        fmt::print("Testing scan loop for 5 seconds...\n");
        int scan_count = 0;
        auto callback = [&scan_count](lidar_manager::PointCloudPtr cloud) {
            scan_count++;
            fmt::print("Received scan #{} with {} points\n", scan_count, cloud->size());
        };

        lidar_acquisition.startScanLoop(callback);
        std::this_thread::sleep_for(std::chrono::seconds(5));
        lidar_acquisition.stopScanLoop();

        fmt::print("Scan loop test completed. Total scans: {}\n", scan_count);
        lidar_acquisition.disconnect();
    } else {
        fmt::print(stderr, "Failed to connect to sensor.\n");
        return 1;
    }

    return 0;
}
