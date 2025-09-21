#include "config.hpp"
#include "cli_parser.hpp"
#include "pcap_reader.hpp"
#include "sensor_info.hpp"
#include "packet_classifier.hpp"
#include "packet_scheduler.hpp"
#include "sender_udp.hpp"
#include "metrics_logger.hpp"

#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>

namespace ouster_sim {

// Global flag for graceful shutdown
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully...\n";
    g_shutdown_requested = true;
}

class OusterSimulator {
public:
    explicit OusterSimulator(const Config& config)
        : config_(config),
          logger_(config.log_path, config.verbose),
          sender_(config.bind_ip),
          scheduler_(config) {
    }

    int run() {
        logger_.log_info("Starting Ouster PCAP simulator");
        logger_.log_info("PCAP file: " + config_.pcap_path);

        // Load sensor info if provided
        std::optional<SensorInfo> sensor_info;
        if (!config_.json_path.empty()) {
            sensor_info = SensorInfo::from_file(config_.json_path);
            if (sensor_info.has_value()) {
                logger_.log_info("Loaded sensor info: " + sensor_info->sensor_name);
                if (config_.verbose) {
                    logger_.log_info(sensor_info->to_string());
                }
            } else {
                logger_.log_warning("Failed to load sensor info from: " + config_.json_path);
            }
        }

        // Initialize classifier
        PacketClassifier classifier(sensor_info);

        // Open PCAP file
        PcapReader reader(config_.pcap_path);
        if (!reader.is_open()) {
            logger_.log_error("Failed to open PCAP file: " + config_.pcap_path);
            return 1;
        }

        // Optional pre-scan
        if (config_.verbose) {
            logger_.log_info("Pre-scanning PCAP file...");
            auto scan_info = reader.pre_scan();
            logger_.log_info("PCAP info: " + std::to_string(scan_info.total_packets) +
                           " packets, " + std::to_string(scan_info.duration_seconds) + "s duration");
            logger_.log_info("Estimated: " + std::to_string(scan_info.lidar_packets) +
                           " LIDAR, " + std::to_string(scan_info.imu_packets) + " IMU packets");
        }

        // Main simulation loop
        uint64_t loop_count = 0;
        do {
            if (config_.loop && loop_count > 0) {
                logger_.log_info("Starting loop iteration " + std::to_string(loop_count + 1));
                reader.reset();
                scheduler_.reset();
            }

            scheduler_.start();
            run_simulation_loop(reader, classifier);

            loop_count++;

        } while (config_.loop && !g_shutdown_requested);

        // Final metrics
        logger_.log_info("Simulation completed");
        log_final_stats(classifier, reader);

        return 0;
    }

private:
    Config config_;
    MetricsLogger logger_;
    SenderUDP sender_;
    PacketScheduler scheduler_;

    void run_simulation_loop(PcapReader& reader, PacketClassifier& classifier) {
        Packet packet;
        uint64_t packets_processed = 0;
        auto last_metrics_time = std::chrono::steady_clock::now();

        while (!g_shutdown_requested && reader.next(packet)) {
            // Classify packet
            PacketType type = classifier.classify(packet);

            // Determine destination
            Endpoint destination;
            switch (type) {
                case PacketType::LIDAR:
                    destination = config_.dst_lidar;
                    break;
                case PacketType::IMU:
                    destination = config_.dst_imu;
                    break;
                case PacketType::UNKNOWN:
                    // Use default based on port or size heuristic
                    if (packet.dst_port == 7502 || packet.payload.size() > 1000) {
                        destination = config_.dst_lidar;
                        type = PacketType::LIDAR;
                    } else {
                        destination = config_.dst_imu;
                        type = PacketType::IMU;
                    }
                    break;
            }

            // Schedule timing
            scheduler_.schedule_packet(packet);

            // Send packet
            bool success = sender_.send_to(destination.ip, destination.port,
                                         packet.payload.data(), packet.payload.size());

            if (success) {
                logger_.log_packet_sent(type, packet.payload.size());
            } else {
                logger_.log_send_error("Failed to send to " + destination.to_string());
            }

            packets_processed++;

            // Periodic metrics
            auto now = std::chrono::steady_clock::now();
            if (now - last_metrics_time >= std::chrono::seconds(5)) {
                logger_.log_periodic_metrics();
                last_metrics_time = now;
            }

            if (config_.verbose && packets_processed % 1000 == 0) {
                logger_.log_info("Processed " + std::to_string(packets_processed) + " packets");
            }
        }
    }

    void log_final_stats(const PacketClassifier& classifier, const PcapReader& reader) {
        logger_.log_info("=== FINAL STATISTICS ===");

        // Reader stats
        logger_.log_info("PCAP: " + std::to_string(reader.packets_read()) +
                        " packets read, " + std::to_string(reader.total_bytes()) + " bytes");

        // Classifier stats
        auto clf_stats = classifier.get_stats();
        logger_.log_info("Classification: " + std::to_string(clf_stats.lidar_packets) +
                        " LIDAR, " + std::to_string(clf_stats.imu_packets) +
                        " IMU, " + std::to_string(clf_stats.unknown_packets) + " unknown");
        if (clf_stats.uncertain_classifications > 0) {
            logger_.log_warning("Uncertain classifications: " +
                              std::to_string(clf_stats.uncertain_classifications));
        }

        // Sender stats
        auto send_stats = sender_.get_stats();
        logger_.log_info("Sender: " + std::to_string(send_stats.packets_sent) +
                        " packets sent, " + std::to_string(send_stats.send_errors) + " errors");

        // Scheduler stats
        auto sched_stats = scheduler_.get_stats();
        logger_.log_info("Scheduler: " + std::to_string(sched_stats.packets_scheduled) +
                        " scheduled, " + std::to_string(sched_stats.zero_deltas) + " zero delays");

        // Final metrics as JSON
        if (config_.verbose) {
            logger_.log_info("Final metrics JSON:\n" + logger_.get_metrics_json());
        }
    }
};

} // namespace ouster_sim

int main(int argc, char* argv[]) {
    try {
        // Set up signal handling
        std::signal(SIGINT, ouster_sim::signal_handler);
        std::signal(SIGTERM, ouster_sim::signal_handler);

        // Parse command line
        ouster_sim::CliParser parser;
        auto config = parser.parse(argc, argv);

        // Create and run simulator
        ouster_sim::OusterSimulator simulator(config);
        return simulator.run();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}