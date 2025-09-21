#include "PointTypes.hpp"
#include <algorithm>
#include <numeric>
#include <limits>

namespace lidar_manager {

PointCloud::PointCloud() {
    points_.reserve(65536); // Réservation par défaut pour un scan typique
}

void PointCloud::addPoint(const PointXYZIRT& point) {
    points_.push_back(point);
    metadata_.points_count = points_.size();
}

void PointCloud::addPoint(float x, float y, float z, uint16_t intensity, uint16_t ring, uint32_t timestamp) {
    points_.emplace_back(x, y, z, intensity, ring, timestamp);
    metadata_.points_count = points_.size();
}

void PointCloud::clear() {
    points_.clear();
    metadata_.points_count = 0;
    metadata_.acquisition_time = std::chrono::high_resolution_clock::now();
}

void PointCloud::reserve(size_t size) {
    points_.reserve(size);
}

void PointCloud::computeBoundingBox(PointXYZIRT& min_point, PointXYZIRT& max_point) const {
    if (points_.empty()) {
        min_point = max_point = PointXYZIRT();
        return;
    }

    min_point.x = max_point.x = points_[0].x;
    min_point.y = max_point.y = points_[0].y;
    min_point.z = max_point.z = points_[0].z;
    min_point.intensity = max_point.intensity = points_[0].intensity;

    for (const auto& point : points_) {
        min_point.x = std::min(min_point.x, point.x);
        min_point.y = std::min(min_point.y, point.y);
        min_point.z = std::min(min_point.z, point.z);
        min_point.intensity = std::min(min_point.intensity, point.intensity);

        max_point.x = std::max(max_point.x, point.x);
        max_point.y = std::max(max_point.y, point.y);
        max_point.z = std::max(max_point.z, point.z);
        max_point.intensity = std::max(max_point.intensity, point.intensity);
    }
}

float PointCloud::computeAverageIntensity() const {
    if (points_.empty()) {
        return 0.0f;
    }

    uint64_t sum = std::accumulate(points_.begin(), points_.end(), 0ULL,
        [](uint64_t acc, const PointXYZIRT& point) {
            return acc + point.intensity;
        });

    return static_cast<float>(sum) / points_.size();
}

} // namespace lidar_manager