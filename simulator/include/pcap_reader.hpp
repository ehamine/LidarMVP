#pragma once

#include "packet.hpp"
#include <memory>
#include <string>

// Forward declaration for pcap_t
struct pcap;
typedef struct pcap pcap_t;

namespace ouster_sim {

class PcapReader {
public:
    explicit PcapReader(const std::string& pcap_path);
    ~PcapReader();

    // Non-copyable, movable
    PcapReader(const PcapReader&) = delete;
    PcapReader& operator=(const PcapReader&) = delete;
    PcapReader(PcapReader&&) noexcept;
    PcapReader& operator=(PcapReader&&) noexcept;

    // Read next packet, returns false when no more packets
    bool next(Packet& packet);

    // Reset to beginning of file (for looping)
    void reset();

    // Get file info
    std::string get_filename() const { return filename_; }
    bool is_open() const { return pcap_handle_ != nullptr; }

    // Statistics
    uint64_t packets_read() const { return packets_read_; }
    uint64_t total_bytes() const { return total_bytes_; }

    // Quick pre-scan to count packets (optional)
    struct ScanInfo {
        uint64_t total_packets;
        uint64_t lidar_packets;
        uint64_t imu_packets;
        double duration_seconds;
    };
    ScanInfo pre_scan();

private:
    std::string filename_;
    pcap_t* pcap_handle_;
    uint64_t packets_read_;
    uint64_t total_bytes_;

    bool open_file();
    void close_file();
    bool parse_ethernet_frame(const uint8_t* data, int len, Packet& packet);
    bool parse_ipv4_packet(const uint8_t* data, int len, Packet& packet);
    bool parse_udp_packet(const uint8_t* data, int len, Packet& packet);
};

} // namespace ouster_sim