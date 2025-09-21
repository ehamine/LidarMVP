#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <cstdint>

namespace lidar_manager {

// Structure pour un point LiDAR avec intensité, ring et timestamp
struct PointXYZIRT {
    float x, y, z;                    // Position 3D (mètres)
    uint16_t intensity;               // Intensité réfléchie (0-65535)
    uint16_t ring;                    // Numéro de l'anneau laser (0-127 pour Ouster)
    uint32_t timestamp;               // Timestamp en microsecondes

    PointXYZIRT() : x(0), y(0), z(0), intensity(0), ring(0), timestamp(0) {}

    PointXYZIRT(float x_, float y_, float z_, uint16_t intensity_,
                uint16_t ring_, uint32_t timestamp_)
        : x(x_), y(y_), z(z_), intensity(intensity_), ring(ring_), timestamp(timestamp_) {}
};

// Métadonnées d'un scan LiDAR
struct ScanMetadata {
    uint64_t scan_id;                 // ID unique du scan
    std::chrono::high_resolution_clock::time_point acquisition_time; // Heure d'acquisition
    uint32_t points_count;            // Nombre de points dans le scan
    float horizontal_resolution;      // Résolution horizontale (degrés)
    float vertical_resolution;        // Résolution verticale (degrés)
    uint16_t lidar_mode;             // Mode du LiDAR (512x10, 1024x10, etc.)

    ScanMetadata() : scan_id(0), points_count(0), horizontal_resolution(0),
                     vertical_resolution(0), lidar_mode(0) {
        acquisition_time = std::chrono::high_resolution_clock::now();
    }
};

// Classe pour un nuage de points
class PointCloud {
public:
    PointCloud();
    ~PointCloud() = default;

    // Gestion des points
    void addPoint(const PointXYZIRT& point);
    void addPoint(float x, float y, float z, uint16_t intensity, uint16_t ring, uint32_t timestamp);
    void clear();
    void reserve(size_t size);

    // Accès aux données
    const std::vector<PointXYZIRT>& getPoints() const { return points_; }
    std::vector<PointXYZIRT>& getPoints() { return points_; }
    size_t size() const { return points_.size(); }
    bool empty() const { return points_.empty(); }

    // Métadonnées
    const ScanMetadata& getMetadata() const { return metadata_; }
    ScanMetadata& getMetadata() { return metadata_; }

    // Opérateurs d'accès
    const PointXYZIRT& operator[](size_t index) const { return points_[index]; }
    PointXYZIRT& operator[](size_t index) { return points_[index]; }

    // Itérateurs
    std::vector<PointXYZIRT>::iterator begin() { return points_.begin(); }
    std::vector<PointXYZIRT>::iterator end() { return points_.end(); }
    std::vector<PointXYZIRT>::const_iterator begin() const { return points_.begin(); }
    std::vector<PointXYZIRT>::const_iterator end() const { return points_.end(); }

    // Statistiques
    void computeBoundingBox(PointXYZIRT& min_point, PointXYZIRT& max_point) const;
    float computeAverageIntensity() const;

private:
    std::vector<PointXYZIRT> points_;
    ScanMetadata metadata_;
};

// Type alias pour faciliter l'utilisation
using PointCloudPtr = std::shared_ptr<PointCloud>;
using PointCloudConstPtr = std::shared_ptr<const PointCloud>;

} // namespace lidar_manager