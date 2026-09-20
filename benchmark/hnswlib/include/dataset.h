#pragma once

/**
 * @file dataset.h
 * @brief Dataset management utilities.
 */

#include <fmt/core.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "file_io.h"

// ============================================================================

namespace hnswlib::benchmark {

enum class Metric { L2, InnerProduct, Cosine };

// Parallel for implementation
template <typename Func>
void parallel_for(size_t start, size_t end, size_t num_threads, Func func) {
    if (num_threads <= 1) {
        for (size_t i = start; i < end; ++i) {
            func(i, 0);
        }
        return;
    }

    std::vector<std::future<void>> futures;
    size_t count = end - start;
    size_t chunk_size = (count + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t t_start = start + t * chunk_size;
        size_t t_end = std::min(end, t_start + chunk_size);

        if (t_start < end) {
            futures.push_back(std::async(std::launch::async, [t_start, t_end, t, func]() {
                for (size_t i = t_start; i < t_end; ++i) {
                    func(i, t);
                }
            }));
        }
    }

    for (auto& f : futures) {
        f.wait();
    }
}

}  // namespace hnswlib::benchmark

namespace hnswlib::benchmark {

// ============================================================================
// Forward Declarations
// ============================================================================

struct DatasetInfo;
class Dataset;

// ============================================================================
// DatasetName - Enum-like class for dataset identification
// ============================================================================

class DatasetName {
public:
    static const DatasetName SIFT1M;
    static const DatasetName DEEP1M;
    static const DatasetName GLOVE;
    static const DatasetName AUDIO;
    static const DatasetName ENRON;
    static const DatasetName ALL;
    static const DatasetName Invalid;

    static const std::array<DatasetName, 5>& all() {
        static const std::array<DatasetName, 5> datasets = {AUDIO, ENRON, SIFT1M, DEEP1M, GLOVE};
        return datasets;
    }

    static DatasetName from_string(const std::string& str) {
        std::string lower = str;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower == ALL.name()) {
            return ALL;
        }

        for (const auto& ds : all()) {
            if (lower == ds.name()) {
                return ds;
            }
        }
        return Invalid;
    }

    const char* name() const { return name_; }
    bool is_valid() const { return name_ != Invalid.name_; }
    const char* to_string() const { return name_; }

    DatasetInfo info() const;

    bool operator==(const DatasetName& other) const { return name_ == other.name_; }
    bool operator!=(const DatasetName& other) const { return name_ != other.name_; }

private:
    constexpr DatasetName(const char* name) : name_(name) {}
    const char* name_;
};

inline constexpr DatasetName DatasetName::SIFT1M{"sift1m"};
inline constexpr DatasetName DatasetName::DEEP1M{"deep1m"};
inline constexpr DatasetName DatasetName::GLOVE{"glove"};
inline constexpr DatasetName DatasetName::AUDIO{"audio"};
inline constexpr DatasetName DatasetName::ENRON{"enron"};
inline constexpr DatasetName DatasetName::ALL{"all"};
inline constexpr DatasetName DatasetName::Invalid{"invalid"};

// ============================================================================
// Dataset Info Structure
// ============================================================================

struct DatasetInfo {
    DatasetName dataset_name;
    std::string download_url;
    Metric metric;
    size_t base_count;
    size_t query_count;
    uint32_t dims;
    uint32_t scale;
    uint32_t explore_depth;

    std::string base_file;
    std::string query_file;
    std::string explore_query_file;
    std::string explore_entry_vertex_file;
    std::string explore_groundtruth_file;
    std::string explore_groundtruth_half_file;
    std::string base_groundtruth_file;
    std::string base_groundtruth_half_file;

    static constexpr size_t EXPLORE_SAMPLE_COUNT = 10000;
    static constexpr uint32_t EXPLORE_TOPK = 1000;
    static constexpr uint32_t GROUNDTRUTH_TOPK = 1024;
    static constexpr size_t GROUNDTRUTH_STEP = 100000;

    const char* name() const { return dataset_name.name(); }
};

inline DatasetInfo make_dataset_info(const DatasetName& ds) {
    DatasetInfo info{ds, {}, Metric::L2, 0, 0, 0, 1, 2, {}, {}, {}, {}, {}, {}, {}, {}};

    std::string name = ds.name();

    info.base_file = name + "_base.fvecs";
    info.query_file = name + "_query.fvecs";
    info.explore_query_file = name + "_explore_query.fvecs";
    info.explore_entry_vertex_file = name + "_explore_entry_vertex.ivecs";
    info.explore_groundtruth_file = name + "_explore_groundtruth_top1000.ivecs";
    info.explore_groundtruth_half_file = name + "_explore_groundtruth_half_top1000.ivecs";
    info.base_groundtruth_file = name + "_base_top1000.ivecs";
    info.base_groundtruth_half_file = name + "_base_half_top1000.ivecs";

    if (ds == DatasetName::SIFT1M) {
        info.base_count = 1000000;
        info.query_count = 10000;
        info.dims = 128;
    } else if (ds == DatasetName::DEEP1M) {
        info.base_count = 1000000;
        info.query_count = 10000;
        info.dims = 96;
        info.scale = 100;
    } else if (ds == DatasetName::GLOVE) {
        info.base_count = 1183514;
        info.query_count = 10000;
        info.dims = 100;
        info.scale = 100;
    } else if (ds == DatasetName::AUDIO) {
        info.base_count = 53387;
        info.query_count = 200;
        info.dims = 192;
        info.explore_depth = 1;
    } else if (ds == DatasetName::ENRON) {
        info.base_count = 94987;
        info.query_count = 200;
        info.dims = 1369;
        info.explore_depth = 1;
    }

    return info;
}

inline DatasetInfo DatasetName::info() const {
    return make_dataset_info(*this);
}

// ============================================================================
// Dataset Class
// ============================================================================

class Dataset {
public:
    Dataset(const DatasetName& name, const std::filesystem::path& data_root)
        : name_(name),
          data_root_(data_root),
          dataset_dir_(data_root / name.name()),
          files_dir_(data_root / name.name() / name.name()),
          info_(name.info()) {}

    const DatasetName& dataset_name() const { return name_; }
    const char* name() const { return name_.name(); }
    bool is_valid() const { return name_.is_valid(); }
    const DatasetInfo& info() const { return info_; }

    const std::filesystem::path& data_root() const { return data_root_; }
    const std::filesystem::path& dataset_dir() const { return dataset_dir_; }
    const std::filesystem::path& files_dir() const { return files_dir_; }

    std::string base_file() const { return (files_dir_ / info_.base_file).string(); }
    std::string query_file() const { return (files_dir_ / info_.query_file).string(); }

    std::string groundtruth_file(size_t nb) const {
        return (files_dir_ / fmt::format("{}_groundtruth_top{}_nb{}.ivecs", info_.name(), DatasetInfo::GROUNDTRUTH_TOPK, nb)).string();
    }
    std::string groundtruth_file_full() const { return groundtruth_file(info_.base_count); }
    std::string groundtruth_file_half() const { return groundtruth_file(info_.base_count / 2); }

    VectorRepository load_base() const { return load_static_repository(base_file().c_str()); }

    VectorRepository load_query() const { return load_static_repository(query_file().c_str()); }

    std::vector<std::vector<uint32_t>> load_groundtruth(size_t k, bool use_half_dataset = false) const {
        std::string gt_file = use_half_dataset ? groundtruth_file_half() : groundtruth_file_full();
        return load_groundtruth_from_file(gt_file, k);
    }

private:
    std::vector<std::vector<uint32_t>> load_groundtruth_from_file(const std::string& gt_file, size_t k) const {
        size_t ground_truth_dims = 0;
        size_t ground_truth_size = 0;
        auto gt_data = ivecs_read(gt_file.c_str(), ground_truth_dims, ground_truth_size);
        const uint32_t* ground_truth = gt_data.get();

        // Fix for potential missing file or empty data check if needed, but assuming file exists as per instructions
        if (!gt_data) {
            fmt::print(stderr, "Could not load ground truth file: {}\n", gt_file);
            std::abort();
        }

        if (ground_truth_dims < k) {
            fmt::print(stderr, "Ground truth data has only {} elements but need {}\n", ground_truth_dims, k);
            std::abort();
        }

        auto answers = std::vector<std::vector<uint32_t>>(ground_truth_size);
        for (size_t i = 0; i < ground_truth_size; i++) {
            auto& gt = answers[i];
            gt.resize(k);
            for (size_t j = 0; j < k; j++) {
                gt[j] = ground_truth[ground_truth_dims * i + j];
            }
            std::sort(gt.begin(), gt.end());
        }

        return answers;
    }

    DatasetName name_;
    std::filesystem::path data_root_;
    std::filesystem::path dataset_dir_;
    std::filesystem::path files_dir_;
    DatasetInfo info_;
};

}  // namespace hnswlib::benchmark
