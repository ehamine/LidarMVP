#include "pcap_reader.hpp"
#include <pcap/pcap.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <net/ethernet.h>
#include <arpa/inet.h>
#include <stdexcept>
#include <cstring>

namespace ouster_sim {

PcapReader::PcapReader(const std::string& pcap_path)
    : filename_(pcap_path), pcap_handle_(nullptr), packets_read_(0), total_bytes_(0) {
    if (!open_file()) {
        throw std::runtime_error("Failed to open pcap file: " + pcap_path);
    }
}

PcapReader::~PcapReader() {
    close_file();
}

PcapReader::PcapReader(PcapReader&& other) noexcept
    : filename_(std::move(other.filename_)),
      pcap_handle_(other.pcap_handle_),
      packets_read_(other.packets_read_),
      total_bytes_(other.total_bytes_) {
    other.pcap_handle_ = nullptr;
    other.packets_read_ = 0;
    other.total_bytes_ = 0;
}

PcapReader& PcapReader::operator=(PcapReader&& other) noexcept {
    if (this != &other) {
        close_file();
        filename_ = std::move(other.filename_);
        pcap_handle_ = other.pcap_handle_;
        packets_read_ = other.packets_read_;
        total_bytes_ = other.total_bytes_;

        other.pcap_handle_ = nullptr;
        other.packets_read_ = 0;
        other.total_bytes_ = 0;
    }
    return *this;
}

bool PcapReader::open_file() {
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_handle_ = pcap_open_offline(filename_.c_str(), errbuf);
    if (!pcap_handle_) {
        return false;
    }
    return true;
}

void PcapReader::close_file() {
    if (pcap_handle_) {
        pcap_close(pcap_handle_);
        pcap_handle_ = nullptr;
    }
}

bool PcapReader::next(Packet& packet) {
    if (!pcap_handle_) {
        return false;
    }

    struct pcap_pkthdr* header;
    const u_char* data;

    int result = pcap_next_ex(pcap_handle_, &header, &data);
    if (result <= 0) {
        return false; // EOF or error
    }

    // Convert timestamp to nanoseconds
    packet.timestamp = std::chrono::nanoseconds(
        static_cast<int64_t>(header->ts.tv_sec) * 1000000000LL +
        static_cast<int64_t>(header->ts.tv_usec) * 1000LL
    );

    // Parse the packet
    if (!parse_ethernet_frame(data, header->caplen, packet)) {
        return false;
    }

    packets_read_++;
    total_bytes_ += header->caplen;

    return true;
}

void PcapReader::reset() {
    if (pcap_handle_) {
        close_file();
        packets_read_ = 0;
        total_bytes_ = 0;
        open_file();
    }
}

bool PcapReader::parse_ethernet_frame(const uint8_t* data, int len, Packet& packet) {
    if (len < sizeof(struct ethhdr)) {
        return false;
    }

    const struct ethhdr* eth_header = reinterpret_cast<const struct ethhdr*>(data);
    uint16_t eth_type = ntohs(eth_header->h_proto);

    // Skip to IP header
    const uint8_t* ip_data = data + sizeof(struct ethhdr);
    int ip_len = len - sizeof(struct ethhdr);

    if (eth_type == ETH_P_IP) {
        return parse_ipv4_packet(ip_data, ip_len, packet);
    }
    // Could add IPv6 support here (ETH_P_IPV6)

    return false;
}

bool PcapReader::parse_ipv4_packet(const uint8_t* data, int len, Packet& packet) {
    if (len < sizeof(struct iphdr)) {
        return false;
    }

    const struct iphdr* ip_header = reinterpret_cast<const struct iphdr*>(data);

    // Check if it's UDP
    if (ip_header->protocol != IPPROTO_UDP) {
        return false;
    }

    // Extract IP addresses
    struct in_addr src_addr, dst_addr;
    src_addr.s_addr = ip_header->saddr;
    dst_addr.s_addr = ip_header->daddr;

    packet.src_ip = inet_ntoa(src_addr);
    packet.dst_ip = inet_ntoa(dst_addr);

    // Calculate IP header length and skip to UDP
    int ip_header_len = ip_header->ihl * 4;
    const uint8_t* udp_data = data + ip_header_len;
    int udp_len = len - ip_header_len;

    return parse_udp_packet(udp_data, udp_len, packet);
}

bool PcapReader::parse_udp_packet(const uint8_t* data, int len, Packet& packet) {
    if (len < sizeof(struct udphdr)) {
        return false;
    }

    const struct udphdr* udp_header = reinterpret_cast<const struct udphdr*>(data);

    packet.src_port = ntohs(udp_header->source);
    packet.dst_port = ntohs(udp_header->dest);

    // Calculate payload
    int udp_header_len = sizeof(struct udphdr);
    int payload_len = len - udp_header_len;

    if (payload_len <= 0) {
        return false;
    }

    // Copy payload
    const uint8_t* payload_data = data + udp_header_len;
    packet.payload.assign(payload_data, payload_data + payload_len);

    return true;
}

PcapReader::ScanInfo PcapReader::pre_scan() {
    ScanInfo info{};

    if (!pcap_handle_) {
        return info;
    }

    // Save current position
    auto current_pos = packets_read_;

    // Reset to beginning
    reset();

    Packet packet;
    auto start_time = std::chrono::nanoseconds::max();
    auto end_time = std::chrono::nanoseconds::min();

    while (next(packet)) {
        info.total_packets++;

        // Update time range
        if (packet.timestamp < start_time) {
            start_time = packet.timestamp;
        }
        if (packet.timestamp > end_time) {
            end_time = packet.timestamp;
        }

        // Simple heuristic classification for pre-scan
        if (packet.dst_port == 7502 || packet.payload.size() > 1200) {
            info.lidar_packets++;
        } else if (packet.dst_port == 7503 || packet.payload.size() <= 200) {
            info.imu_packets++;
        }
    }

    if (start_time != std::chrono::nanoseconds::max() &&
        end_time != std::chrono::nanoseconds::min()) {
        info.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
    }

    // Reset to beginning for actual processing
    reset();

    return info;
}

} // namespace ouster_sim