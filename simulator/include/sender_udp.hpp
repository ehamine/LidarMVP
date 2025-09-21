#pragma once

#include "packet.hpp"
#include "config.hpp"
#include <string>
#include <chrono>

namespace ouster_sim {

class SenderUDP {
public:
    explicit SenderUDP(const std::string& bind_ip = "0.0.0.0");
    ~SenderUDP();

    // Non-copyable, movable
    SenderUDP(const SenderUDP&) = delete;
    SenderUDP& operator=(const SenderUDP&) = delete;
    SenderUDP(SenderUDP&&) noexcept;
    SenderUDP& operator=(SenderUDP&&) noexcept;

    // Send packet to specified destination
    bool send_to(const std::string& dst_ip, int dst_port,
                 const uint8_t* data, size_t size);

    // Send packet using destination from packet
    bool send_packet(const Packet& packet, const Endpoint& override_dst);

    // Configuration
    void set_buffer_size(int size);
    void set_tos(int tos);

    // Statistics
    struct Stats {
        uint64_t packets_sent{0};
        uint64_t bytes_sent{0};
        uint64_t send_errors{0};
        std::chrono::nanoseconds total_send_time{0};
        std::chrono::nanoseconds min_send_time{std::chrono::nanoseconds::max()};
        std::chrono::nanoseconds max_send_time{0};
    };

    const Stats& get_stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

    // Check if socket is valid
    bool is_valid() const { return socket_fd_ >= 0; }

private:
    int socket_fd_;
    std::string bind_ip_;
    mutable Stats stats_;

    bool create_socket();
    void close_socket();
    void update_send_stats(bool success, size_t bytes,
                          std::chrono::nanoseconds duration) const;
};

} // namespace ouster_sim