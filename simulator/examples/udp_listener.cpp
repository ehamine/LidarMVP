// Simple UDP packet listener for testing ouster_sim
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <signal.h>
#include <atomic>
#include <iomanip>

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "\nShutting down...\n";
    g_running = false;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <port>\n";
        std::cout << "Example: " << argv[0] << " 7502\n";
        return 1;
    }

    int port = std::atoi(argv[1]);
    if (port <= 0 || port > 65535) {
        std::cerr << "Invalid port: " << port << std::endl;
        return 1;
    }

    signal(SIGINT, signal_handler);

    // Create UDP socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return 1;
    }

    // Bind to port
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sock);
        return 1;
    }

    std::cout << "UDP listener started on port " << port << std::endl;
    std::cout << "Press Ctrl+C to stop\n\n";

    // Statistics
    uint64_t packet_count = 0;
    uint64_t total_bytes = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_report = start_time;

    char buffer[65536];
    struct sockaddr_in sender_addr;
    socklen_t sender_len = sizeof(sender_addr);

    while (g_running) {
        // Set receive timeout
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        ssize_t bytes_received = recvfrom(sock, buffer, sizeof(buffer), 0,
                                        (struct sockaddr*)&sender_addr, &sender_len);

        if (bytes_received > 0) {
            packet_count++;
            total_bytes += bytes_received;

            auto now = std::chrono::steady_clock::now();

            // Report every 5 seconds
            if (now - last_report >= std::chrono::seconds(5)) {
                auto elapsed = std::chrono::duration<double>(now - start_time).count();
                double pps = packet_count / elapsed;
                double mbps = (total_bytes / elapsed) / (1024.0 * 1024.0);

                std::cout << "Packets: " << packet_count
                         << ", Rate: " << std::fixed << std::setprecision(1) << pps << " pps"
                         << ", Throughput: " << mbps << " MB/s"
                         << ", Last packet: " << bytes_received << " bytes from "
                         << inet_ntoa(sender_addr.sin_addr) << ":" << ntohs(sender_addr.sin_port)
                         << std::endl;

                last_report = now;
            }
        }
    }

    close(sock);

    // Final report
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n=== Final Statistics ===\n";
    std::cout << "Total packets: " << packet_count << std::endl;
    std::cout << "Total bytes: " << total_bytes << std::endl;
    std::cout << "Duration: " << total_elapsed << " seconds\n";
    std::cout << "Average rate: " << (packet_count / total_elapsed) << " pps\n";
    std::cout << "Average throughput: " << ((total_bytes / total_elapsed) / (1024.0 * 1024.0)) << " MB/s\n";

    return 0;
}