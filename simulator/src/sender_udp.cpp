#include "sender_udp.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <stdexcept>

namespace ouster_sim {

SenderUDP::SenderUDP(const std::string& bind_ip)
    : socket_fd_(-1), bind_ip_(bind_ip) {
    if (!create_socket()) {
        throw std::runtime_error("Failed to create UDP socket");
    }
}

SenderUDP::~SenderUDP() {
    close_socket();
}

SenderUDP::SenderUDP(SenderUDP&& other) noexcept
    : socket_fd_(other.socket_fd_), bind_ip_(std::move(other.bind_ip_)),
      stats_(other.stats_) {
    other.socket_fd_ = -1;
}

SenderUDP& SenderUDP::operator=(SenderUDP&& other) noexcept {
    if (this != &other) {
        close_socket();
        socket_fd_ = other.socket_fd_;
        bind_ip_ = std::move(other.bind_ip_);
        stats_ = other.stats_;
        other.socket_fd_ = -1;
    }
    return *this;
}

bool SenderUDP::create_socket() {
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        return false;
    }

    // Set socket buffer size (larger for better performance)
    int buffer_size = 1024 * 1024;  // 1MB
    setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF,
               &buffer_size, sizeof(buffer_size));

    // Bind to specified IP if provided
    if (bind_ip_ != "0.0.0.0") {
        struct sockaddr_in bind_addr;
        memset(&bind_addr, 0, sizeof(bind_addr));
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_addr.s_addr = inet_addr(bind_ip_.c_str());
        bind_addr.sin_port = 0;  // Any port

        if (bind(socket_fd_, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
            close_socket();
            return false;
        }
    }

    return true;
}

void SenderUDP::close_socket() {
    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }
}

bool SenderUDP::send_to(const std::string& dst_ip, int dst_port,
                       const uint8_t* data, size_t size) {
    if (socket_fd_ < 0 || size == 0) {
        update_send_stats(false, 0, std::chrono::nanoseconds::zero());
        return false;
    }

    auto start_time = std::chrono::steady_clock::now();

    // Create destination address
    struct sockaddr_in dest_addr;
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(dst_port);

    if (inet_aton(dst_ip.c_str(), &dest_addr.sin_addr) == 0) {
        update_send_stats(false, 0, std::chrono::nanoseconds::zero());
        return false;
    }

    // Send packet
    ssize_t bytes_sent = sendto(socket_fd_, data, size, 0,
                               (struct sockaddr*)&dest_addr, sizeof(dest_addr));

    auto end_time = std::chrono::steady_clock::now();
    auto duration = end_time - start_time;

    bool success = (bytes_sent == static_cast<ssize_t>(size));
    update_send_stats(success, success ? size : 0, duration);

    return success;
}

bool SenderUDP::send_packet(const Packet& packet, const Endpoint& override_dst) {
    return send_to(override_dst.ip, override_dst.port,
                   packet.payload.data(), packet.payload.size());
}

void SenderUDP::set_buffer_size(int size) {
    if (socket_fd_ >= 0) {
        setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size));
    }
}

void SenderUDP::set_tos(int tos) {
    if (socket_fd_ >= 0) {
        setsockopt(socket_fd_, IPPROTO_IP, IP_TOS, &tos, sizeof(tos));
    }
}

void SenderUDP::update_send_stats(bool success, size_t bytes,
                                 std::chrono::nanoseconds duration) const {
    if (success) {
        stats_.packets_sent++;
        stats_.bytes_sent += bytes;
        stats_.total_send_time += duration;
        stats_.min_send_time = std::min(stats_.min_send_time, duration);
        stats_.max_send_time = std::max(stats_.max_send_time, duration);
    } else {
        stats_.send_errors++;
    }
}

} // namespace ouster_sim