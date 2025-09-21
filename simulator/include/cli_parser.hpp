#pragma once

#include "config.hpp"
#include <vector>
#include <string>

namespace ouster_sim {

class CliParser {
public:
    CliParser();

    // Parse command line arguments
    Config parse(int argc, char* argv[]);

    // Print help message
    void print_help(const std::string& program_name) const;

    // Print version information
    void print_version() const;

private:
    struct Option {
        std::string short_name;
        std::string long_name;
        std::string description;
        bool has_value;
        bool required;
    };

    std::vector<Option> options_;

    void init_options();
    bool parse_endpoint(const std::string& str, Endpoint& endpoint) const;
    void validate_config(const Config& config) const;
};

} // namespace ouster_sim