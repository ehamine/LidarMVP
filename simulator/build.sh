#!/bin/bash

set -e

# Build script for Ouster PCAP Simulator

echo "=== Ouster PCAP Simulator Build Script ==="

# Check dependencies
echo "Checking dependencies..."

if ! pkg-config --exists libpcap; then
    echo "Error: libpcap not found. Install with:"
    echo "  sudo apt-get install libpcap-dev"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found. Install with:"
    echo "  sudo apt-get install cmake"
    exit 1
fi

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executables:"
echo "  ./ouster_sim - Main simulator"
echo "  ./examples/udp_listener - UDP packet listener for testing"
echo "  ./examples/mock_pcap_generator - Generate test PCAP files"

if [ -f "test_ouster_sim" ]; then
    echo "  ./test_ouster_sim - Unit tests"
fi

echo ""
echo "Quick test:"
echo "  1. ./examples/mock_pcap_generator test.pcap"
echo "  2. ./ouster_sim --pcap test.pcap --verbose"
echo ""
echo "For help: ./ouster_sim --help"