#pragma once

/**
 * @file build.h
 * @brief Shared helpers for HNSW benchmarks (deglib-style).
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "benchmark.h"
#include "dataset.h"
#include "hnswlib.h"
#include "logging.h"
#include "stats.h"

namespace hnswlib::benchmark {

// -----------------------------------------------------------------------------
// Test configurations (shared)
// -----------------------------------------------------------------------------

struct CreateGraphTest {
    size_t M = 16;
    size_t maxM0 = 32;
    size_t ef_construction = 200;
    size_t seed = 7;
    uint32_t build_threads = 1;

    uint32_t anns_k = 100;
    uint32_t anns_repeat = 1;
    uint32_t anns_threads = 1;

    uint32_t explore_k = 1000;
    uint32_t explore_repeat = 1;
    uint32_t explore_threads = 1;

    std::vector<float> ef_parameter = {10, 20, 40, 60, 80, 100, 200, 400};
};

struct ThreadScalingTest {
    std::vector<uint32_t> thread_counts = {1, 2, 4, 8, 16, 32};
};

enum class DynamicScenario { AddHalf, AddAllRemoveHalf, AddHalfRemoveAndAddOneAtATime };

inline const char* dynamic_scenario_str(DynamicScenario scenario) {
    switch (scenario) {
        case DynamicScenario::AddHalf:
            return "AddHalf";
        case DynamicScenario::AddAllRemoveHalf:
            return "AddAllRemoveHalf";
        case DynamicScenario::AddHalfRemoveAndAddOneAtATime:
            return "AddHalfRemoveAndAddOneAtATime";
        default:
            return "Unknown";
    }
}

struct DynamicDataTest {
    std::vector<DynamicScenario> scenarios = {
        DynamicScenario::AddHalf, DynamicScenario::AddAllRemoveHalf, DynamicScenario::AddHalfRemoveAndAddOneAtATime};
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

inline void wait_before_test(int seconds = 10) {
    log("Waiting {} seconds for machine to settle...\n", seconds);
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

inline std::unique_ptr<SpaceInterface<float>> create_space(const Dataset& ds, size_t dims) {
    if (ds.info().metric == Metric::L2) {
        return std::make_unique<L2Space>(dims);
    }
    return std::make_unique<InnerProductSpace>(dims);
}

inline std::vector<std::vector<uint32_t>> load_ivecs_as_vectors(const char* filename, size_t& count) {
    size_t d = 0, n = 0;
    auto ptr = ivecs_read(filename, d, n);
    count = n;

    std::vector<std::vector<uint32_t>> res(n);
    if (!ptr) return res;

    for (size_t i = 0; i < n; ++i) {
        res[i].assign(ptr.get() + i * d, ptr.get() + (i + 1) * d);
    }
    return res;
}

template <typename SpaceT>
inline std::unique_ptr<HierarchicalNSW<float>> build_index(SpaceT* space,
                                                           const VectorRepository& base_repo,
                                                           size_t M,
                                                           size_t ef_construction,
                                                           size_t maxM0,
                                                           size_t seed,
                                                           uint32_t threads,
                                                           size_t count = 0,
                                                           size_t offset = 0) {
    const size_t max_elements = base_repo.size();
    if (offset > max_elements) {
        offset = max_elements;
    }

    const size_t requested_end = (count == 0) ? max_elements : std::min(max_elements, offset + count);
    const size_t build_count = (requested_end > offset) ? (requested_end - offset) : 0;
    const size_t start_label = offset;
    const size_t end_label = requested_end;
    auto index = std::make_unique<HierarchicalNSW<float>>(space, max_elements, M, ef_construction, seed, maxM0);

    if (build_count == 0) return index;

    log("Building index: {} elements\n", build_count);

    index->addPoint(base_repo.getFeature(start_label), start_label);

    StopW stopw;
    StopW stopw_full;

    // Progress reporting (roughly ~10 lines for large builds).
    const size_t report_every = std::max<size_t>(10000, build_count / 10);

    auto report_progress = [&](size_t done) {
        if (report_every == 0) return;

        const auto elapsed_us = static_cast<double>(stopw.getElapsedTimeMicro());
        const double kips = (elapsed_us > 0.0) ? (static_cast<double>(report_every) * 1000.0 / elapsed_us) : 0.0;
        const double seconds_total = 1e-6 * static_cast<double>(stopw_full.getElapsedTimeMicro());
        const double percent = 100.0 * static_cast<double>(done) / static_cast<double>(build_count);
        const size_t mem_mb = getCurrentRSS() / 1000000;

        log("{:.2f} %, {:.2f} kips {:.2f}s  Mem: {} Mb\n", percent, kips, seconds_total, mem_mb);
        stopw.reset();
    };

    if (threads <= 1) {
        size_t done = 1;
        for (size_t label = start_label + 1; label < end_label; ++label) {
            index->addPoint(base_repo.getFeature(label), label);
            ++done;
            if (report_every != 0 && (done % report_every) == 0) {
                report_progress(done);
            }
        }
    } else {
        std::atomic<size_t> inserted{1};
        std::atomic<size_t> next_report{report_every};
        std::mutex report_mutex;

        parallel_for(start_label + 1, end_label, threads, [&](size_t label, size_t) {
            index->addPoint(base_repo.getFeature(label), label);

            if (report_every == 0) return;

            const size_t done = ++inserted;
            size_t target = next_report.load(std::memory_order_relaxed);
            if (done >= target && target != 0) {
                std::lock_guard<std::mutex> lock(report_mutex);
                target = next_report.load(std::memory_order_relaxed);
                if (done >= target && target != 0) {
                    report_progress(done);
                    next_report.store(target + report_every, std::memory_order_relaxed);
                }
            }
        });
    }

    log("Build time: {:.2f} seconds. Mem: {} Mb. Peak Mem: {} Mb.\n",
        1e-6 * static_cast<double>(stopw_full.getElapsedTimeMicro()),
        getCurrentRSS() / 1000000,
        getPeakRSS() / 1000000);

    return index;
}

template <typename Index>
inline void run_graph_stats(Index* index, const Dataset& ds, uint32_t feature_dims, bool use_half_gt) {
    const std::string gt_file = use_half_gt ? (ds.files_dir() / ds.info().base_groundtruth_half_file).string()
                                            : (ds.files_dir() / ds.info().base_groundtruth_file).string();
    if (std::filesystem::exists(gt_file)) {
        stats::compute_stats(index, gt_file.c_str(), feature_dims);
    } else {
        log("Skipping stats: ground truth file not found {}\n", gt_file);
    }
}

template <typename Index>
inline void run_anns_test(
    Index* index, const VectorRepository& query_repo, const Dataset& ds, const CreateGraphTest& cg, bool use_half_gt) {
    auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
    wait_before_test();

    // ef must be >= k for meaningful TOP-k recall evaluation.
    std::vector<float> ef_filtered;
    ef_filtered.reserve(cg.ef_parameter.size());
    for (float ef : cg.ef_parameter) {
        if (ef >= static_cast<float>(cg.anns_k)) {
            ef_filtered.push_back(ef);
        }
    }
    if (ef_filtered.empty()) {
        ef_filtered.push_back(static_cast<float>(cg.anns_k));
    }

    test_graph_anns(index, query_repo, ground_truth, cg.anns_repeat, cg.anns_threads, cg.anns_k, ef_filtered);
}

template <typename Index>
inline void run_explore_test(Index* index, const Dataset& ds, const CreateGraphTest& cg, bool use_half_gt) {
    std::string entry_file = (ds.files_dir() / ds.info().explore_entry_vertex_file).string();
    const std::string explore_gt_file = use_half_gt ? (ds.files_dir() / ds.info().explore_groundtruth_half_file).string()
                                                    : (ds.files_dir() / ds.info().explore_groundtruth_file).string();

    if (std::filesystem::exists(entry_file) && std::filesystem::exists(explore_gt_file)) {
        size_t entry_count = 0;
        auto entry_indices = load_ivecs_as_vectors(entry_file.c_str(), entry_count);

        size_t dim_gt = 0, n_gt = 0;
        auto gt_ptr = ivecs_read(explore_gt_file.c_str(), dim_gt, n_gt);
        std::vector<std::vector<uint32_t>> explore_gt_vec(n_gt);
        if (gt_ptr) {
            for (size_t i = 0; i < n_gt; ++i) {
                explore_gt_vec[i].assign(gt_ptr.get() + i * dim_gt, gt_ptr.get() + (i + 1) * dim_gt);
                std::sort(explore_gt_vec[i].begin(), explore_gt_vec[i].end());
            }
            wait_before_test();
            test_explore(index, explore_gt_vec, entry_indices, cg.explore_k);
        }
    } else {
        log("Skipping exploration test: missing files\n");
        log("Entry: {}\n", entry_file);
    }
}

template <typename Index>
inline void run_common_tests(
    Index* index, const Dataset& ds, const VectorRepository& query_repo, const CreateGraphTest& cg, bool use_half_gt) {
    run_graph_stats(index, ds, static_cast<uint32_t>(ds.info().dims), use_half_gt);

    log("\n--- ANNS Test (k={}) ---\n", cg.anns_k);
    run_anns_test(index, query_repo, ds, cg, use_half_gt);

    log("\n--- Exploration Test (k={}) ---\n", cg.explore_k);
    run_explore_test(index, ds, cg, use_half_gt);
}

}  // namespace hnswlib::benchmark
