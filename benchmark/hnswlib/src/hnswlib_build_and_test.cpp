/**
 * @file hnswlib_build_and_test.cpp
 * @brief HNSW benchmark tool modeled after deglib_build_and_test.cpp.
 */

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "build.h"
#include "dataset.h"

using namespace hnswlib;
using namespace hnswlib::benchmark;

struct DatasetConfig {
    DatasetName dataset_name = DatasetName::SIFT1M;
    Metric metric = Metric::L2;

    CreateGraphTest create_graph;
    ThreadScalingTest thread_scaling_test;
    DynamicDataTest dynamic_data_test;
};

static DatasetConfig get_dataset_config(const DatasetName& dataset_name) {
    DatasetConfig conf{};
    conf.dataset_name = dataset_name;
    conf.create_graph.seed = 7;

    if (dataset_name == DatasetName::SIFT1M) {
        conf.create_graph.M = 40;
        conf.create_graph.maxM0 = 50;
        conf.create_graph.ef_construction = 800;
        conf.create_graph.ef_parameter = {100, 120, 150, 200, 300, 600};
    } else if (dataset_name == DatasetName::DEEP1M) {
        conf.create_graph.M = 40;
        conf.create_graph.maxM0 = 50;
        conf.create_graph.ef_construction = 800;
        conf.create_graph.ef_parameter = {100, 140, 171, 206, 249, 500, 1000};
    } else if (dataset_name == DatasetName::GLOVE) {
        conf.create_graph.M = 50;
        conf.create_graph.maxM0 = 60;
        conf.create_graph.ef_construction = 700;
        conf.create_graph.ef_parameter = {1000, 1500, 2000, 2500, 5000, 10000, 20000, 40000, 80000};
    } else if (dataset_name == DatasetName::ENRON) {
        conf.create_graph.M = 10;
        conf.create_graph.maxM0 = 50;
        conf.create_graph.ef_construction = 700;
        conf.create_graph.ef_parameter = {100, 125, 150, 200, 300, 600};
    } else if (dataset_name == DatasetName::AUDIO) {
        conf.create_graph.M = 10;
        conf.create_graph.maxM0 = 50;
        conf.create_graph.ef_construction = 700;
        conf.create_graph.ef_parameter = {100, 125, 150, 200, 300, 600};
    } else {
        // Fallback: keep CreateGraphTest defaults but make sure ef sweep is sane for k=100.
        conf.create_graph.ef_parameter = {100, 120, 150, 200, 300, 600};
    }

    return conf;
}

// -----------------------------------------------------------------------------
// Graph path utilities (deglib-style directories/logs)
// -----------------------------------------------------------------------------

struct GraphPaths {
    std::filesystem::path graph_dir;

    GraphPaths(const Dataset& ds) : graph_dir(ds.data_root() / ds.name() / "hnsw") {}

    static std::string metric_str(Metric m) {
        switch (m) {
            case Metric::L2:
                return "L2";
            case Metric::InnerProduct:
                return "IP";
            case Metric::Cosine:
                return "Cosine";
            default:
                return "Unknown";
        }
    }

    std::string base_name(uint32_t dims, Metric metric, size_t M, size_t maxM0, size_t ef_construction) const {
        return fmt::format("{}D_{}_M{}_maxM0{}_Ef{}_HNSW", dims, metric_str(metric), M, maxM0, ef_construction);
    }

    std::string graph_directory() const { return graph_dir.string(); }

    std::string graph_file(uint32_t dims, Metric metric, size_t M, size_t maxM0, size_t ef_construction) const {
        return (graph_dir / (base_name(dims, metric, M, maxM0, ef_construction) + ".hnsw")).string();
    }

    std::string graph_log_file(uint32_t dims, Metric metric, size_t M, size_t maxM0, size_t ef_construction) const {
        return (graph_dir / (base_name(dims, metric, M, maxM0, ef_construction) + ".log")).string();
    }

    std::string thread_scaling_directory() const { return (graph_dir / "threadScaling").string(); }
    std::string dynamic_directory() const { return (graph_dir / "dynamic").string(); }

    std::string thread_scaling_graph_file(
        uint32_t dims, Metric metric, size_t M, size_t maxM0, size_t ef_construction, uint32_t threads) const {
        return (std::filesystem::path(thread_scaling_directory()) /
                fmt::format("{}_T{}.hnsw", base_name(dims, metric, M, maxM0, ef_construction), threads))
            .string();
    }

    std::string thread_scaling_log_file(
        uint32_t dims, Metric metric, size_t M, size_t maxM0, size_t ef_construction, uint32_t threads) const {
        return (std::filesystem::path(thread_scaling_directory()) /
                fmt::format("{}_T{}.log", base_name(dims, metric, M, maxM0, ef_construction), threads))
            .string();
    }

    std::string dynamic_graph_file(
        uint32_t dims, Metric metric, size_t M, size_t maxM0, size_t ef_construction, const std::string& scenario) const {
        return (std::filesystem::path(dynamic_directory()) /
                fmt::format("{}_{}.hnsw", base_name(dims, metric, M, maxM0, ef_construction), scenario))
            .string();
    }

    std::string dynamic_log_file(
        uint32_t dims, Metric metric, size_t M, size_t maxM0, size_t ef_construction, const std::string& scenario) const {
        return (std::filesystem::path(dynamic_directory()) /
                fmt::format("{}_{}.log", base_name(dims, metric, M, maxM0, ef_construction), scenario))
            .string();
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

static void run_create_graph_test(const Dataset& ds,
                                  const DatasetConfig& config,
                                  const GraphPaths& paths,
                                  const VectorRepository& base_repo,
                                  const VectorRepository& query_repo,
                                  bool force_test) {
    const auto& cg = config.create_graph;
    const uint32_t dims = static_cast<uint32_t>(base_repo.dims());

    std::string graph_path = paths.graph_file(dims, config.metric, cg.M, cg.maxM0, cg.ef_construction);
    std::string log_path = paths.graph_log_file(dims, config.metric, cg.M, cg.maxM0, cg.ef_construction);

    log("\n=== CREATE_GRAPH Test ===\n");
    log("Settings: M={}, maxM0={}, ef_construction={}, seed={}, threads={}\n",
        cg.M,
        cg.maxM0,
        cg.ef_construction,
        cg.seed,
        cg.build_threads);
    log("Graph: {}\n", graph_path);
    log("Log: {}\n", log_path);

    std::filesystem::create_directories(paths.graph_directory());
    bool exists = std::filesystem::exists(log_path);
    if (!force_test && exists) {
        log("CREATE_GRAPH: Skipping - log file already exists: {}\n", log_path);
        return;
    }
    set_log_file(log_path, force_test && exists);

    auto space = create_space(ds, dims);

    std::unique_ptr<HierarchicalNSW<float>> index;

    if (std::filesystem::exists(graph_path)) {
        log("Graph already exists, loading: {}\n", graph_path);
        index = std::make_unique<HierarchicalNSW<float>>(space.get(), graph_path, false, base_repo.size());
    } else {
        log("\n--- Building graph ---\n");
        index = build_index(space.get(), base_repo, cg.M, cg.ef_construction, cg.maxM0, cg.seed, cg.build_threads);

        index->saveIndex(graph_path);
        log("Saved graph: {}\n", graph_path);
    }

    if (index) {
        run_common_tests(index.get(), ds, query_repo, cg, false);
    }

    reset_log_to_console();
    log("CREATE_GRAPH: Log written to: {}\n", log_path);
}

static void run_thread_scaling_test(const Dataset& ds,
                                    const DatasetConfig& config,
                                    const GraphPaths& paths,
                                    const VectorRepository& base_repo,
                                    const VectorRepository& query_repo,
                                    bool force_test) {
    const auto& ts = config.thread_scaling_test;
    const auto& cg = config.create_graph;
    const uint32_t dims = static_cast<uint32_t>(base_repo.dims());

    std::string scaling_dir = paths.thread_scaling_directory();
    log("\n=== THREAD_SCALING Test ===\n");
    log("Testing thread counts: [{}]\n", fmt::join(ts.thread_counts, ", "));
    log("Settings: M={}, maxM0={}, ef_construction={}, seed={}\n", cg.M, cg.maxM0, cg.ef_construction, cg.seed);
    log("Directory: {}\n", scaling_dir);

    std::filesystem::create_directories(scaling_dir);

    for (uint32_t threads : ts.thread_counts) {
        std::string graph_path = paths.thread_scaling_graph_file(dims, config.metric, cg.M, cg.maxM0, cg.ef_construction, threads);
        std::string log_path = paths.thread_scaling_log_file(dims, config.metric, cg.M, cg.maxM0, cg.ef_construction, threads);

        bool exists = std::filesystem::exists(log_path);
        if (!force_test && exists) {
            log("threads={}: Skipping - log file already exists: {}\n", threads, log_path);
            continue;
        }
        set_log_file(log_path, force_test && exists);
        log("\n=== THREAD_SCALING Test: threads={} ===\n", threads);
        log("Graph: {}\n", graph_path);

        auto space = create_space(ds, dims);

        std::unique_ptr<HierarchicalNSW<float>> index;
        if (std::filesystem::exists(graph_path)) {
            log("Graph already exists, loading: {}\n", graph_path);
            index = std::make_unique<HierarchicalNSW<float>>(space.get(), graph_path, false, base_repo.size());
        } else {
            log("\n--- Building graph with {} threads ---\n", threads);
            index = build_index(space.get(), base_repo, cg.M, cg.ef_construction, cg.maxM0, cg.seed, threads);
            index->saveIndex(graph_path);
            log("Saved graph: {}\n", graph_path);
        }

        if (index) {
            log("\n--- ANNS Test (k={}) ---\n", cg.anns_k);
            run_anns_test(index.get(), query_repo, ds, cg, false);
        }

        reset_log_to_console();
        log("threads={}: Log written to: {}\n", threads, log_path);
    }
}

static void run_dynamic_data_test(const Dataset& ds,
                                  const DatasetConfig& config,
                                  const GraphPaths& paths,
                                  const VectorRepository& base_repo,
                                  const VectorRepository& query_repo,
                                  bool force_test) {
    const auto& cg = config.create_graph;
    const auto& dd = config.dynamic_data_test;
    const uint32_t dims = static_cast<uint32_t>(base_repo.dims());

    std::string dynamic_dir = paths.dynamic_directory();
    std::filesystem::create_directories(dynamic_dir);

    for (auto scenario : dd.scenarios) {
        std::string scenario_name = dynamic_scenario_str(scenario);
        std::string graph_path = paths.dynamic_graph_file(dims, config.metric, cg.M, cg.maxM0, cg.ef_construction, scenario_name);
        std::string log_path = paths.dynamic_log_file(dims, config.metric, cg.M, cg.maxM0, cg.ef_construction, scenario_name);

        log("\n=== DYNAMIC_DATA Test: {} ===\n", scenario_name);
        log("Settings: M={}, maxM0={}, ef_construction={}, seed={}, threads={}\n",
            cg.M,
            cg.maxM0,
            cg.ef_construction,
            cg.seed,
            cg.build_threads);
        log("Graph: {}\n", graph_path);
        log("Log: {}\n", log_path);

        bool exists = std::filesystem::exists(log_path);
        if (!force_test && exists) {
            log("{}: Skipping - log file already exists: {}\n", scenario_name, log_path);
            continue;
        }
        set_log_file(log_path, force_test && exists);

        auto space = create_space(ds, dims);

        std::unique_ptr<HierarchicalNSW<float>> index;
        const size_t max_elements = base_repo.size();
        const size_t half_elements = max_elements / 2;

        if (scenario == DynamicScenario::AddHalf) {
            index = build_index(space.get(), base_repo, cg.M, cg.ef_construction, cg.maxM0, cg.seed, cg.build_threads, half_elements);
        } else if (scenario == DynamicScenario::AddAllRemoveHalf) {
            index = build_index(space.get(), base_repo, cg.M, cg.ef_construction, cg.maxM0, cg.seed, cg.build_threads);

            log("\n--- Dynamic updates ---\n");
            log("Deleting {} elements...\n", half_elements);
            StopW del_stopw;
            for (size_t i = half_elements; i < max_elements; ++i) {
                index->markDelete(i);
            }
            log("Delete time: {:.2f}s\n", del_stopw.getElapsedTimeMicro() / 1e6);
        } else if (scenario == DynamicScenario::AddHalfRemoveAndAddOneAtATime) {
            // IMPORTANT: The half-dataset ground truth files correspond to the first half of labels [0..half-1].
            // For this scenario we want to end up with exactly that active set.
            // Therefore: start with the SECOND half in the index, then swap it out one-by-one.
            const size_t second_half_begin = half_elements;
            const size_t second_half_count = max_elements - half_elements;

            index = build_index(space.get(),
                                base_repo,
                                cg.M,
                                cg.ef_construction,
                                cg.maxM0,
                                cg.seed,
                                cg.build_threads,
                                /*count=*/second_half_count,
                                /*offset=*/second_half_begin);

            log("\n--- Dynamic updates ---\n");
            StopW update_stopw;
            for (size_t label = 0; label < half_elements; ++label) {
                index->markDelete(label + half_elements);
                index->addPoint(base_repo.getFeature(label), label);
            }

            // If max_elements is odd, the second half has one extra element that isn't covered by the
            // one-to-one swap above (labels [half .. 2*half-1]). Delete any remaining second-half labels.
            for (size_t label = 2 * half_elements; label < max_elements; ++label) {
                index->markDelete(label);
            }
            log("Update time: {:.2f}s\n", update_stopw.getElapsedTimeMicro() / 1e6);
        }

        if (index) {
            index->saveIndex(graph_path);
            run_common_tests(index.get(), ds, query_repo, cg, true);
        }

        reset_log_to_console();
        log("{}: Log written to: {}\n", scenario_name, log_path);
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

static void run_for_dataset(
    const DatasetName& ds_name, const std::string& data_root, const std::string& test_type_arg, bool do_run, bool force_test) {
    Dataset ds(ds_name, data_root);
    auto config = get_dataset_config(ds_name);
    config.metric = ds.info().metric;
    GraphPaths graph_paths(ds);

    log("\n=== Dataset: {} ===\n", ds.name());
    log("Repository file: {}\n", ds.base_file());
    log("Query file: {}\n", ds.query_file());
    log("Graph directory: {}\n", graph_paths.graph_directory());
    log("Ground truth (full): {}\n", ds.groundtruth_file_full());
    log("Ground truth (half): {}\n", ds.groundtruth_file_half());
    log("Metric: {}\n", GraphPaths::metric_str(config.metric));
    log("Build settings: M={}, maxM0={}, ef_construction={}, seed={}, threads={}\n",
        config.create_graph.M,
        config.create_graph.maxM0,
        config.create_graph.ef_construction,
        config.create_graph.seed,
        config.create_graph.build_threads);

    if (do_run) {
        if (!std::filesystem::exists(ds.base_file())) {
            log("Dataset files not found in {}.\n", ds.files_dir().string());
            return;
        }

        log("\nLoading data...\n");
        auto base_repo = ds.load_base();
        auto query_repo = ds.load_query();
        log("Loaded {} base vectors and {} queries\n", base_repo.size(), query_repo.size());

        bool run_all = (test_type_arg == "all");

        if (run_all || test_type_arg == "create_graph") {
            run_create_graph_test(ds, config, graph_paths, base_repo, query_repo, force_test);
        }

        if (run_all || test_type_arg == "thread_scaling") {
            run_thread_scaling_test(ds, config, graph_paths, base_repo, query_repo, force_test);
        }

        if (run_all || test_type_arg == "dynamic_data") {
            run_dynamic_data_test(ds, config, graph_paths, base_repo, query_repo, force_test);
        }
    }
}

int main(int argc, char** argv) {
    log("Testing ...\n");

#if defined(USE_AVX)
    log("use AVX2  ...\n");
#elif defined(USE_SSE)
    log("use SSE  ...\n");
#else
    log("use arch  ...\n");
#endif

    const auto data_path = std::filesystem::path(DATA_PATH);
    log("data_path {} \n", data_path.string());

    DatasetName ds_name = DatasetName::ALL;
    std::string test_type_arg = "all";
    std::string data_root = data_path.string();
    bool do_run = true;
    bool force_test = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "help" || arg == "--help") {
            log("Usage: hnsw_benchmark <dataset> [test_type] [data_root] [--run|--dry-run] [--force-test]\n");
            log("Datasets: sift1m, deep1m, audio, glove, enron, all\n");
            log("Test types:\n");
            log("  create_graph    - Build graph, run stats, ANNS, explore\n");
            log("  thread_scaling  - Build graphs with different thread counts\n");
            log("  dynamic_data    - Build graphs for dynamic scenarios (AddHalf, AddAllRemoveHalf, AddHalfRemoveAndAddOneAtATime)\n");
            log("  all             - Run all available tests\n");
            log("Options: [data_root] path (default: DATA_PATH), --run or --dry-run, --force-test\n");
            return 0;
        }

        if (arg == "--run") {
            do_run = true;
            continue;
        }
        if (arg == "--dry-run") {
            do_run = false;
            continue;
        }
        if (arg == "--force-test") {
            force_test = true;
            continue;
        }

        auto parsed_ds = DatasetName::from_string(arg);
        if (parsed_ds.is_valid() || parsed_ds == DatasetName::ALL) {
            ds_name = parsed_ds;
            continue;
        }

        if (arg == "create_graph" || arg == "thread_scaling" || arg == "dynamic_data" || arg == "all") {
            test_type_arg = arg;
            continue;
        }

        data_root = arg;
    }

    if (ds_name == DatasetName::ALL) {
        for (const auto& ds : DatasetName::all()) {
            run_for_dataset(ds, data_root, test_type_arg, do_run, force_test);
        }
    } else {
        run_for_dataset(ds_name, data_root, test_type_arg, do_run, force_test);
    }

    log("\nTest OK\n");
    return 0;
}
