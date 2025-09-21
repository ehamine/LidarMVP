#include "cli_parser.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <sstream>

namespace ouster_sim {

CliParser::CliParser() {
    init_options();
}

void CliParser::init_options() {
    options_ = {
        {"p", "pcap", "PCAP file path (required)", true, true},
        {"j", "json", "Sensor info JSON file path", true, false},
        {"", "dst-lidar", "Destination for LIDAR packets (ip:port)", true, false},
        {"", "dst-imu", "Destination for IMU packets (ip:port)", true, false},
        {"r", "rate", "Playback rate multiplier (default: 1.0)", true, false},
        {"l", "loop", "Loop playback", false, false},
        {"", "no-timestamps", "Ignore pcap timestamps, use fixed rate", false, false},
        {"", "jitter", "Add random jitter in seconds (default: 0.0)", true, false},
        {"", "max-delta", "Maximum sleep between packets in seconds (default: 1.0)", true, false},
        {"b", "bind", "Bind IP address (default: 0.0.0.0)", true, false},
        {"v", "verbose", "Verbose logging", false, false},
        {"", "log", "Log file path", true, false},
        {"", "metrics-port", "HTTP port for metrics (0=disabled)", true, false},
        {"h", "help", "Show this help message", false, false},
        {"", "version", "Show version information", false, false}
    };
}

Config CliParser::parse(int argc, char* argv[]) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            exit(0);
        } else if (arg == "--version") {
            print_version();
            exit(0);
        } else if (arg == "-p" || arg == "--pcap") {
            if (++i >= argc) throw std::runtime_error("--pcap requires a value");
            config.pcap_path = argv[i];
        } else if (arg == "-j" || arg == "--json") {
            if (++i >= argc) throw std::runtime_error("--json requires a value");
            config.json_path = argv[i];
        } else if (arg == "--dst-lidar") {
            if (++i >= argc) throw std::runtime_error("--dst-lidar requires a value");
            if (!parse_endpoint(argv[i], config.dst_lidar)) {
                throw std::runtime_error("Invalid --dst-lidar format. Use ip:port");
            }
        } else if (arg == "--dst-imu") {
            if (++i >= argc) throw std::runtime_error("--dst-imu requires a value");
            if (!parse_endpoint(argv[i], config.dst_imu)) {
                throw std::runtime_error("Invalid --dst-imu format. Use ip:port");
            }
        } else if (arg == "-r" || arg == "--rate") {
            if (++i >= argc) throw std::runtime_error("--rate requires a value");
            config.rate = std::stod(argv[i]);
            if (config.rate <= 0.0) {
                throw std::runtime_error("--rate must be positive");
            }
        } else if (arg == "-l" || arg == "--loop") {
            config.loop = true;
        } else if (arg == "--no-timestamps") {
            config.no_timestamps = true;
        } else if (arg == "--jitter") {
            if (++i >= argc) throw std::runtime_error("--jitter requires a value");
            config.jitter = std::stod(argv[i]);
            if (config.jitter < 0.0) {
                throw std::runtime_error("--jitter must be non-negative");
            }
        } else if (arg == "--max-delta") {
            if (++i >= argc) throw std::runtime_error("--max-delta requires a value");
            config.max_delta = std::stod(argv[i]);
            if (config.max_delta <= 0.0) {
                throw std::runtime_error("--max-delta must be positive");
            }
        } else if (arg == "-b" || arg == "--bind") {
            if (++i >= argc) throw std::runtime_error("--bind requires a value");
            config.bind_ip = argv[i];
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--log") {
            if (++i >= argc) throw std::runtime_error("--log requires a value");
            config.log_path = argv[i];
        } else if (arg == "--metrics-port") {
            if (++i >= argc) throw std::runtime_error("--metrics-port requires a value");
            config.metrics_http_port = std::stoi(argv[i]);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    validate_config(config);
    return config;
}

bool CliParser::parse_endpoint(const std::string& str, Endpoint& endpoint) const {
    size_t colon_pos = str.find(':');
    if (colon_pos == std::string::npos) {
        return false;
    }

    endpoint.ip = str.substr(0, colon_pos);
    try {
        endpoint.port = std::stoi(str.substr(colon_pos + 1));
        return endpoint.port > 0 && endpoint.port <= 65535;
    } catch (...) {
        return false;
    }
}

void CliParser::validate_config(const Config& config) const {
    if (config.pcap_path.empty()) {
        throw std::runtime_error("--pcap is required");
    }

    if (!config.is_valid()) {
        throw std::runtime_error("Invalid configuration");
    }
}

void CliParser::print_help(const std::string& program_name) const {
    std::cout << "Ouster PCAP Simulator\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Required:\n";
    std::cout << "  -p, --pcap PATH        PCAP file to replay\n\n";

    std::cout << "Network Options:\n";
    std::cout << "  -j, --json PATH        Sensor info JSON file\n";
    std::cout << "      --dst-lidar IP:PORT  Destination for LIDAR packets (default: 127.0.0.1:7502)\n";
    std::cout << "      --dst-imu IP:PORT    Destination for IMU packets (default: 127.0.0.1:7503)\n";
    std::cout << "  -b, --bind IP          Bind socket to IP address (default: 0.0.0.0)\n\n";

    std::cout << "Timing Options:\n";
    std::cout << "  -r, --rate FLOAT       Playback rate multiplier (default: 1.0)\n";
    std::cout << "  -l, --loop             Loop playback\n";
    std::cout << "      --no-timestamps    Ignore pcap timestamps, use fixed rate\n";
    std::cout << "      --jitter FLOAT     Add random jitter in seconds (default: 0.0)\n";
    std::cout << "      --max-delta FLOAT  Maximum sleep between packets (default: 1.0)\n\n";

    std::cout << "Logging Options:\n";
    std::cout << "  -v, --verbose          Verbose logging\n";
    std::cout << "      --log PATH         Log file path\n";
    std::cout << "      --metrics-port PORT  HTTP port for metrics (0=disabled)\n\n";

    std::cout << "Other:\n";
    std::cout << "  -h, --help             Show this help message\n";
    std::cout << "      --version          Show version information\n\n";

    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --pcap sample.pcap --json sensor_info.json\n";
    std::cout << "  " << program_name << " --pcap data.pcap --rate 2.0 --loop --verbose\n";
    std::cout << "  " << program_name << " --pcap test.pcap --dst-lidar 192.168.1.10:7502\n";
}

void CliParser::print_version() const {
    std::cout << "ouster_sim version 0.1.0\n";
    std::cout << "Built with libpcap support\n";
}

} // namespace ouster_sim