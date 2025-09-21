# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LibarMVP is a C++20 CMake project implementing an enhanced LiDAR data acquisition and processing system, specifically designed for Ouster sensors. The project is in Phase 1 of development, focusing on basic LiDAR acquisition and data processing capabilities.

## Build System and Dependencies

The project uses **CMake 3.24+** with **Conan** for dependency management. Key dependencies include:
- **fmt**: String formatting and logging
- **OusterSDK**: Official Ouster sensor SDK for data acquisition
- **GTest**: Unit testing framework

### Build Commands

```bash
# Install dependencies
conan install . --build=missing

# Configure and build
cmake --preset conan-default
cmake --build --preset conan-release

# Or use direct cmake approach:
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Test Commands

```bash
# Run all tests
ctest

# Run specific test binary
./build/tests/run_tests
```

## Architecture

### Core Components

1. **LidarAcquisition** (`src/LidarAcquisition.cpp`, `include/LidarAcquisition.hpp`)
   - Main interface for sensor connection and data acquisition
   - Manages Ouster sensor client lifecycle
   - Uses custom deleter pattern for Ouster SDK resources

2. **OusterDriverEnhanced** (`src/OusterDriverEnhanced.cpp`, `include/OusterDriverEnhanced.hpp`)
   - Enhanced wrapper around Ouster SDK functionality
   - Designed for advanced features (currently placeholder implementations):
     - Network quality monitoring
     - Packet loss detection
     - Auto-calibration management
     - Sensor drift detection

3. **Main Application** (`src/main.cpp`)
   - CLI entry point requiring sensor hostname parameter
   - Basic connection testing and lifecycle management

### Build Targets

- **lidar_manager_core**: Static library containing core functionality
- **lidar_manager**: Main executable
- **run_tests**: Test runner using GTest

### Development Notes

- The project is currently in **Phase 1**: basic acquisition infrastructure
- Many advanced features in `OusterDriverEnhanced` are commented out placeholders
- Uses forward declarations to minimize header dependencies and compilation time
- Follows modern C++20 practices with RAII and smart pointers

### CMake Build Options

- `BUILD_WITH_CUDA`: Enable CUDA support (default: OFF)
- `BUILD_WITH_TENSORRT`: Enable TensorRT (default: OFF)
- `BUILD_SECURITY_FEATURES`: Enable security features (default: OFF)
- `BUILD_MONITORING`: Enable monitoring (default: OFF)
- `ENABLE_LTO`: Enable Link Time Optimization (default: ON)

### Platform Detection

The build system includes automatic platform detection via `cmake/DetectPlatform.cmake`.

### Usage

```bash
# Main application (requires real sensor)
./build/src/lidar_manager <sensor_hostname>

# Scan acquisition demo (offline)
./build/src/scan_demo

# Architecture performance test (offline)
./build/src/offline_test
```

The main application expects an Ouster sensor hostname/IP. Demo applications work offline for testing.

### Current Capabilities

**Phase 1 - Completed:**
- Real-time LiDAR data acquisition (getNextScan, startScanLoop)
- Point cloud data structures (PointXYZIRT, PointCloud, ScanMetadata)
- Enhanced driver integration with network monitoring
- Thread-safe continuous acquisition with callbacks
- Comprehensive testing suite with mock data generation
- Performance: 32,768 points/scan @ 9.5 Hz

**Key Files:**
- `include/PointTypes.hpp` - Point cloud data structures
- `src/LidarAcquisition.cpp` - Main acquisition implementation
- `test_*.cpp` - Demo applications and tests

**Next Steps:** See `NEXT_STEPS.md` for detailed roadmap to Phase 2 (Advanced Processing).