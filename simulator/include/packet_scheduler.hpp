#pragma once

#include "packet.hpp"
#include "config.hpp"
#include <chrono>
#include <random>
#include <optional>

namespace ouster_sim {

class PacketScheduler {
public:
    explicit PacketScheduler(const Config& config);

    // Initialize timing for first packet
    void start();

    // Calculate and sleep for the appropriate delay before sending packet
    void schedule_packet(const Packet& packet);

    // Reset for looping
    void reset();

    // Statistics
    struct Stats {
        uint64_t packets_scheduled{0};
        std::chrono::nanoseconds total_sleep_time{0};
        std::chrono::nanoseconds min_delta{std::chrono::nanoseconds::max()};
        std::chrono::nanoseconds max_delta{0};
        uint64_t zero_deltas{0};
        uint64_t capped_deltas{0};
    };

    const Stats& get_stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    Config config_;
    mutable Stats stats_;

    // Timing state
    std::optional<std::chrono::nanoseconds> last_packet_time_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point wallclock_start_;

    // Jitter simulation
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<double> jitter_dist_;

    // Helper methods
    std::chrono::nanoseconds calculate_sleep_duration(const Packet& packet);
    void precise_sleep(std::chrono::nanoseconds duration);
    double get_jitter() const;
};

} // namespace ouster_sim