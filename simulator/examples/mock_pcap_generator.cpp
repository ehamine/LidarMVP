// Generate a mock PCAP file for testing purposes
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <arpa/inet.h>

#pragma pack(push, 1)

// PCAP file header
struct pcap_hdr_t {
    uint32_t magic_number;   /* magic number */
    uint16_t version_major;  /* major version number */
    uint16_t version_minor;  /* minor version number */
    int32_t  thiszone;       /* GMT to local correction */
    uint32_t sigfigs;        /* accuracy of timestamps */
    uint32_t snaplen;        /* max length of captured packets, in octets */
    uint32_t network;        /* data link type */
};

// PCAP record header
struct pcaprec_hdr_t {
    uint32_t ts_sec;         /* timestamp seconds */
    uint32_t ts_usec;        /* timestamp microseconds */
    uint32_t incl_len;       /* number of octets of packet saved in file */
    uint32_t orig_len;       /* actual length of packet */
};

// Ethernet header
struct ethhdr {
    uint8_t  h_dest[6];      /* destination eth addr */
    uint8_t  h_source[6];    /* source ether addr */
    uint16_t h_proto;        /* packet type ID field */
};

// IP header
struct iphdr {
    uint8_t  ihl:4;          /* header length */
    uint8_t  version:4;      /* version */
    uint8_t  tos;            /* type of service */
    uint16_t tot_len;        /* total length */
    uint16_t id;             /* identification */
    uint16_t frag_off;       /* fragment offset field */
    uint8_t  ttl;            /* time to live */
    uint8_t  protocol;       /* protocol */
    uint16_t check;          /* checksum */
    uint32_t saddr;          /* source address */
    uint32_t daddr;          /* dest address */
};

// UDP header
struct udphdr {
    uint16_t source;         /* source port */
    uint16_t dest;           /* destination port */
    uint16_t len;            /* udp length */
    uint16_t check;          /* udp checksum */
};

#pragma pack(pop)

class MockPcapGenerator {
public:
    MockPcapGenerator(const std::string& filename) : filename_(filename) {}

    bool generate(int num_lidar_packets = 1000, int num_imu_packets = 100) {
        std::ofstream file(filename_, std::ios::binary);
        if (!file) {
            return false;
        }

        // Write PCAP header
        write_pcap_header(file);

        auto start_time = std::chrono::system_clock::now();
        auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch()).count();

        // Generate packets with realistic timing
        for (int i = 0; i < num_lidar_packets; ++i) {
            // LIDAR packets every ~100µs (10kHz)
            write_udp_packet(file, time_us + i * 100, 7502, generate_lidar_payload());
        }

        for (int i = 0; i < num_imu_packets; ++i) {
            // IMU packets every 10ms (100Hz)
            write_udp_packet(file, time_us + i * 10000, 7503, generate_imu_payload());
        }

        std::cout << "Generated " << filename_ << " with "
                  << num_lidar_packets << " LIDAR and "
                  << num_imu_packets << " IMU packets\n";

        return true;
    }

private:
    std::string filename_;

    void write_pcap_header(std::ofstream& file) {
        pcap_hdr_t header = {};
        header.magic_number = 0xa1b2c3d4;
        header.version_major = 2;
        header.version_minor = 4;
        header.thiszone = 0;
        header.sigfigs = 0;
        header.snaplen = 65536;
        header.network = 1; // Ethernet

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    }

    void write_udp_packet(std::ofstream& file, uint64_t timestamp_us,
                         uint16_t dst_port, const std::vector<uint8_t>& payload) {
        // Calculate packet sizes
        size_t payload_size = payload.size();
        size_t udp_size = sizeof(udphdr) + payload_size;
        size_t ip_size = sizeof(iphdr) + udp_size;
        size_t eth_size = sizeof(ethhdr) + ip_size;

        // Write packet record header
        pcaprec_hdr_t rec_header = {};
        rec_header.ts_sec = static_cast<uint32_t>(timestamp_us / 1000000);
        rec_header.ts_usec = static_cast<uint32_t>(timestamp_us % 1000000);
        rec_header.incl_len = static_cast<uint32_t>(eth_size);
        rec_header.orig_len = static_cast<uint32_t>(eth_size);

        file.write(reinterpret_cast<const char*>(&rec_header), sizeof(rec_header));

        // Write Ethernet header
        ethhdr eth = {};
        memset(eth.h_dest, 0xaa, 6);
        memset(eth.h_source, 0xbb, 6);
        eth.h_proto = htons(0x0800); // IPv4

        file.write(reinterpret_cast<const char*>(&eth), sizeof(eth));

        // Write IP header
        iphdr ip = {};
        ip.version = 4;
        ip.ihl = 5;
        ip.tot_len = htons(static_cast<uint16_t>(ip_size));
        ip.ttl = 64;
        ip.protocol = 17; // UDP
        ip.saddr = inet_addr("192.168.1.100");
        ip.daddr = inet_addr("192.168.1.200");

        file.write(reinterpret_cast<const char*>(&ip), sizeof(ip));

        // Write UDP header
        udphdr udp = {};
        udp.source = htons(12345);
        udp.dest = htons(dst_port);
        udp.len = htons(static_cast<uint16_t>(udp_size));

        file.write(reinterpret_cast<const char*>(&udp), sizeof(udp));

        // Write payload
        file.write(reinterpret_cast<const char*>(payload.data()), payload_size);
    }

    std::vector<uint8_t> generate_lidar_payload() {
        // Generate realistic LIDAR packet payload (simplified)
        std::vector<uint8_t> payload(1456); // Typical Ouster packet size

        // Fill with some pattern
        for (size_t i = 0; i < payload.size(); ++i) {
            payload[i] = static_cast<uint8_t>(i % 256);
        }

        return payload;
    }

    std::vector<uint8_t> generate_imu_payload() {
        // Generate realistic IMU packet payload
        std::vector<uint8_t> payload(48); // Typical IMU packet size

        // Fill with some pattern
        for (size_t i = 0; i < payload.size(); ++i) {
            payload[i] = static_cast<uint8_t>(0x10 + (i % 16));
        }

        return payload;
    }
};

int main(int argc, char* argv[]) {
    std::string filename = "test.pcap";
    int lidar_packets = 1000;
    int imu_packets = 100;

    if (argc > 1) filename = argv[1];
    if (argc > 2) lidar_packets = std::atoi(argv[2]);
    if (argc > 3) imu_packets = std::atoi(argv[3]);

    std::cout << "Generating mock PCAP file: " << filename << std::endl;

    MockPcapGenerator generator(filename);
    if (!generator.generate(lidar_packets, imu_packets)) {
        std::cerr << "Failed to generate PCAP file" << std::endl;
        return 1;
    }

    std::cout << "Done! Test with:" << std::endl;
    std::cout << "  ./ouster_sim --pcap " << filename << " --verbose" << std::endl;

    return 0;
}