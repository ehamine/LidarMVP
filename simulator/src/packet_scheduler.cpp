#include "packet_scheduler.hpp"
#include <thread>
#include <algorithm>
#include <ctime>

namespace ouster_sim {

PacketScheduler::PacketScheduler(const Config& config)
    : config_(config), rng_(std::random_device{}()), jitter_dist_(0.0, config.jitter) {
}

void PacketScheduler::start() {
    start_time_ = std::chrono::steady_clock::now();
    wallclock_start_ = start_time_;
    last_packet_time_.reset();
    reset_stats();
}

void PacketScheduler::schedule_packet(const Packet& packet) {
    auto sleep_duration = calculate_sleep_duration(packet);

    if (sleep_duration > std::chrono::nanoseconds::zero()) {
        // Add jitter if configured
        if (config_.jitter > 0.0) {
            auto jitter_ns = std::chrono::nanoseconds(
                static_cast<int64_t>(get_jitter() * 1e9)
            );
            sleep_duration += jitter_ns;
        }

        // Cap the sleep duration if max_delta is set
        auto max_sleep = std::chrono::nanoseconds(
            static_cast<int64_t>(config_.max_delta * 1e9)
        );
        if (sleep_duration > max_sleep) {
            sleep_duration = max_sleep;
            stats_.capped_deltas++;
        }

        precise_sleep(sleep_duration);

        // Update stats
        stats_.total_sleep_time += sleep_duration;
        stats_.min_delta = std::min(stats_.min_delta, sleep_duration);
        stats_.max_delta = std::max(stats_.max_delta, sleep_duration);
    } else {
        stats_.zero_deltas++;
    }

    stats_.packets_scheduled++;
    last_packet_time_ = packet.timestamp;
}

void PacketScheduler::reset() {
    last_packet_time_.reset();
    // Don't reset start_time_ to maintain continuous timing across loops
}

std::chrono::nanoseconds PacketScheduler::calculate_sleep_duration(const Packet& packet) {
    if (config_.no_timestamps) {
        // Fixed interval mode - estimate based on common scan rates
        // Typical Ouster: 10-20 Hz, 512-2048 packets per scan
        // Assume 10 Hz, 512 packets = ~19.5 μs per packet
        return std::chrono::microseconds(20);
    }

    if (!last_packet_time_.has_value()) {
        // First packet - no delay needed
        return std::chrono::nanoseconds::zero();
    }

    // Calculate time delta from pcap
    auto pcap_delta = packet.timestamp - last_packet_time_.value();

    // Apply rate scaling
    auto scaled_delta = std::chrono::nanoseconds(
        static_cast<int64_t>(pcap_delta.count() / config_.rate)
    );

    if (config_.align_wallclock) {
        // Align to wall clock time
        // Calculate elapsed time in the original pcap
        auto pcap_elapsed = packet.timestamp - (last_packet_time_.has_value() ?
            last_packet_time_.value() : packet.timestamp);

        auto expected_time = wallclock_start_ +
            std::chrono::nanoseconds(static_cast<int64_t>(
                pcap_elapsed.count() / config_.rate
            ));
        auto current_time = std::chrono::steady_clock::now();
        return expected_time - current_time;
    }

    return scaled_delta;
}

void PacketScheduler::precise_sleep(std::chrono::nanoseconds duration) {
    if (duration <= std::chrono::nanoseconds::zero()) {
        return;
    }

#ifdef __linux__
    // Use clock_nanosleep for better precision on Linux
    struct timespec ts;
    ts.tv_sec = duration.count() / 1000000000LL;
    ts.tv_nsec = duration.count() % 1000000000LL;

    clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, nullptr);
#else
    // Fallback to std::this_thread::sleep_for
    std::this_thread::sleep_for(duration);
#endif
}

double PacketScheduler::get_jitter() const {
    if (config_.jitter <= 0.0) {
        return 0.0;
    }
    return jitter_dist_(rng_);
}

} // namespace ouster_sim