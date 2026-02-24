#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <boost/program_options.hpp>

#include "benchmark.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <iostream>

#include "index.h"
#include "index_factory.h"
#include "dataset.h"

using namespace diskann::benchmark;

struct DiskANNBuildParams
{
    uint32_t R = 64;
    uint32_t L = 100;
    float alpha = 1.2f;
    uint32_t build_PQ_bytes = 0;
    bool use_opq = false;
    uint32_t max_occlusion_size = 750;
};

struct DatasetConfig
{
    DatasetName dataset_name = DatasetName::SIFT1M;
    DiskANNBuildParams build_params;

    // ANNS Test Params
    uint32_t anns_k = 100;
    uint32_t anns_repeat = 1;
    std::vector<uint32_t> Lvec = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    // Exploration Params
    uint32_t explore_k = 1000;
    float explore_recall_target = 0.995f;
};

static DatasetConfig get_dataset_config(const DatasetName &dataset_name)
{
    DatasetConfig conf;
    conf.dataset_name = dataset_name;

    if (dataset_name == DatasetName::SIFT1M)
    {
        conf.build_params.R = 64;
        conf.build_params.L = 100;
        conf.build_params.alpha = 1.2f;
        conf.Lvec = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200};
    }
    else if (dataset_name == DatasetName::DEEP1M)
    {
        conf.build_params.R = 64;
        conf.build_params.L = 100;
        conf.build_params.alpha = 1.2f;
        conf.anns_k = 100;
        conf.Lvec = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200};
    }
    else if (dataset_name == DatasetName::GLOVE)
    {
        conf.build_params.R = 64;
        conf.build_params.L = 100;
        conf.build_params.alpha = 1.2f;
        conf.anns_k = 100;
        conf.Lvec = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200};
    }
    else if (dataset_name == DatasetName::AUDIO)
    {
        conf.build_params.R = 64;
        conf.build_params.L = 100;
        conf.build_params.alpha = 1.2f;
        conf.anns_k = 20;
        conf.anns_repeat = 5;
        conf.explore_k = 1000;
        conf.Lvec = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    }
    else if (dataset_name == DatasetName::ENRON)
    {
        conf.build_params.R = 64;
        conf.build_params.L = 100;
        conf.build_params.alpha = 1.2f;
        conf.anns_k = 20;
        conf.explore_k = 1000;
        conf.Lvec = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    }

    // Filter Lvec to ensure L >= anns_k
    std::vector<uint32_t> filtered_Lvec;
    for (uint32_t L : conf.Lvec)
    {
        if (L >= conf.anns_k)
        {
            filtered_Lvec.push_back(L);
        }
    }
    conf.Lvec = filtered_Lvec;

    return conf;
}

// namespace po = boost::program_options; // Removed
std::string get_index_path(const Dataset &ds, const DatasetConfig &conf)
{
    std::string prefix = ds.dataset_dir().string() + "/diskann/diskann_R" + std::to_string(conf.build_params.R) + "_L" +
                         std::to_string(conf.build_params.L);
    return prefix;
}

void run_create_index(const std::string &index_path, const Dataset &ds, const DatasetConfig &conf, uint32_t num_threads)
{
    log("Building DiskANN index: %s\n", index_path.c_str());
    auto build_params = conf.build_params;

    std::string data_type = "float";
    std::string dist_fn = ds.info().metric == diskann::Metric::L2
                              ? "l2"
                              : (ds.info().metric == diskann::Metric::INNER_PRODUCT ? "mips" : "cosine");

    size_t data_num = ds.info().base_count;
    size_t data_dim = ds.info().dims;
    size_t aligned_dim = ROUND_UP(data_dim, 8);

    auto data_wrapper = ds.load_base();
    float *data = data_wrapper.data;

    std::vector<uint32_t> tags(data_num);
    std::iota(tags.begin(), tags.end(), 1); // tag 0 is reserved for hidden points

    auto index_build_params = diskann::IndexWriteParametersBuilder(build_params.L, build_params.R)
                                  .with_max_occlusion_size(build_params.max_occlusion_size)
                                  .with_filter_list_size(0)
                                  .with_alpha(build_params.alpha)
                                  .with_num_threads(num_threads)
                                  .build();

    auto index_search_params =
        diskann::IndexSearchParams(index_build_params.search_list_size, index_build_params.num_threads);

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(ds.info().metric)
                      .with_dimension(ds.info().dims)
                      .with_max_points(ds.info().base_count)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type("float")
                      .with_label_type("uint")
                      .with_index_write_params(index_build_params)
                      .with_index_search_params(index_search_params)
                      .is_dynamic_index(true)
                      .is_enable_tags(true)
                      .is_use_opq(build_params.use_opq)
                      .is_pq_dist_build(build_params.build_PQ_bytes > 0)
                      .with_num_pq_chunks(build_params.build_PQ_bytes)
                      .is_concurrent_consolidate(false)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->set_start_points_at_random(static_cast<float>(0));

    StopW timer;

    for (size_t i = 0; i < data_num; i++)
    {
        index->insert_point(&data[i * aligned_dim], tags[i]);
        if (i > 0 && i % 100000 == 0)
        {
            log("added %zu after %.2f seconds.\n", i, (timer.getElapsedTimeMicro() / 1000000.0));
        }
    }

    log("added %zu after %.2f seconds.\n", data_num, (timer.getElapsedTimeMicro() / 1000000.0));

    // Save dynamic index
    index->save(index_path.c_str(), true);
}

enum class DynamicScenario
{
    AddHalf,
    AddAllRemoveHalf,
    AddHalfRemoveAndAddOneAtATime
};

inline const char *dynamic_scenario_str(DynamicScenario scenario)
{
    switch (scenario)
    {
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

void run_dynamic_data_test(const Dataset &ds, const DatasetConfig &conf, bool force_test, uint32_t num_threads)
{
    auto build_params = conf.build_params;
    size_t data_num = ds.info().base_count;
    size_t data_dim = ds.info().dims;
    size_t aligned_dim = ROUND_UP(data_dim, 8);
    auto data_wrapper = ds.load_base();
    float *data = data_wrapper.data;

    std::vector<uint32_t> tags(data_num);
    std::iota(tags.begin(), tags.end(), 1);

    auto query_data = ds.load_query();
    size_t query_num = ds.info().query_count;
    size_t query_dim = ds.info().dims;
    size_t query_aligned_dim = ROUND_UP(query_dim, 8);

    auto ground_truth = ds.load_groundtruth(conf.anns_k, true);

    std::vector<DynamicScenario> scenarios = {DynamicScenario::AddHalf, DynamicScenario::AddAllRemoveHalf,
                                              DynamicScenario::AddHalfRemoveAndAddOneAtATime};

    for (auto scenario : scenarios)
    {
        std::string scenario_name = dynamic_scenario_str(scenario);
        std::string index_path = ds.dataset_dir().string() + "/diskann/dynamic/diskann_R" +
                                 std::to_string(conf.build_params.R) + "_L" + std::to_string(conf.build_params.L) +
                                 "_" + scenario_name + ".da";
        std::string log_file = index_path + "_benchmark.log";

        ensure_directory(ds.dataset_dir() / "diskann" / "dynamic");

        log("\n=== DYNAMIC_DATA Test: %s ===\n", scenario_name.c_str());
        log("Settings: R=%u, L=%u, threads=%u\n", build_params.R, build_params.L, num_threads);

        if (!force_test && diskann::benchmark::file_exists(log_file))
        {
            log("%s: Skipping - log file already exists: %s\n", scenario_name.c_str(), log_file.c_str());
            continue;
        }

        set_log_file(log_file, force_test && diskann::benchmark::file_exists(log_file));
        attach_cout_to_log();

        try
        {
            auto index_build_params = diskann::IndexWriteParametersBuilder(build_params.L, build_params.R)
                                          .with_max_occlusion_size(750)
                                          .with_filter_list_size(0)
                                          .with_alpha(build_params.alpha)
                                          .with_num_threads(num_threads)
                                          .build();

            auto index_search_params =
                diskann::IndexSearchParams(index_build_params.search_list_size, index_build_params.num_threads);

            auto config = diskann::IndexConfigBuilder()
                              .with_metric(ds.info().metric)
                              .with_dimension(ds.info().dims)
                              .with_max_points(ds.info().base_count)
                              .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                              .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                              .with_data_type("float")
                              .with_label_type("uint")
                              .with_index_write_params(index_build_params)
                              .with_index_search_params(index_search_params)
                              .is_dynamic_index(true)
                              .is_enable_tags(true)
                              .is_use_opq(build_params.use_opq)
                              .is_pq_dist_build(build_params.build_PQ_bytes > 0)
                              .with_num_pq_chunks(build_params.build_PQ_bytes)
                              .is_concurrent_consolidate(false)
                              .build();

            auto index_factory = diskann::IndexFactory(config);
            auto index = index_factory.create_instance();
            index->set_start_points_at_random(static_cast<float>(0));

            const size_t max_elements = data_num;
            const size_t half_elements = max_elements / 2;

            StopW timer;
            log("\n--- Dynamic updates ---\n");

            if (scenario == DynamicScenario::AddHalf)
            {
                for (size_t i = 0; i < half_elements; ++i)
                {
                    index->insert_point(&data[i * aligned_dim], tags[i]);
                }
                log("Update time: %.2f s\n", (timer.getElapsedTimeMicro() / 1e6));
            }
            else if (scenario == DynamicScenario::AddAllRemoveHalf)
            {
                for (size_t i = 0; i < max_elements; ++i)
                {
                    index->insert_point(&data[i * aligned_dim], tags[i]);
                }
                StopW del_stopw;
                for (size_t i = half_elements; i < max_elements; ++i)
                {
                    index->lazy_delete(tags[i]);
                }
                log("Delete time: %.2f s\n", (del_stopw.getElapsedTimeMicro() / 1e6));
            }
            else if (scenario == DynamicScenario::AddHalfRemoveAndAddOneAtATime)
            {
                for (size_t i = half_elements; i < max_elements; ++i)
                {
                    index->insert_point(&data[i * aligned_dim], tags[i]);
                }
                StopW update_stopw;
                for (size_t i = 0; i < half_elements; ++i)
                {
                    index->lazy_delete(tags[i + half_elements]);
                    index->insert_point(&data[i * aligned_dim], tags[i]);
                }
                for (size_t i = 2 * half_elements; i < max_elements; ++i)
                {
                    index->lazy_delete(tags[i]);
                }
                log("Update time: %.2f s\n", (update_stopw.getElapsedTimeMicro() / 1e6));
            }

            if (scenario != DynamicScenario::AddHalf)
            {
                StopW cons_stopw;
                index->consolidate_deletes(index_build_params);
                log("Consolidate time: %.2f s\n", (cons_stopw.getElapsedTimeMicro() / 1e6));
            }

            index->save(index_path.c_str(), true);

            // Test the index in memory directly without reload
            auto typed_index = dynamic_cast<diskann::Index<float, uint32_t, uint32_t> *>(index.get());
            if (typed_index)
            {
                if (ds.info().metric == diskann::FAST_L2)
                {
                    typed_index->optimize_index_layout();
                }
                log("\n--- ANNS Test (k=%u) ---\n", conf.anns_k);
                test_diskann_anns<float, uint32_t, uint32_t>(typed_index, query_data.data, query_num, query_dim,
                                                             query_aligned_dim, ground_truth, conf.anns_k, conf.Lvec,
                                                             num_threads);
            }

            log("%s: Log written to: %s\n", scenario_name.c_str(), log_file.c_str());
        }
        catch (const std::exception &e)
        {
            log("Exception in dynamic test '%s': %s\n", scenario_name.c_str(), e.what());
        }

        detach_cout_from_log();
        reset_log_to_console();
    }
}

void run_explore_test(diskann::AbstractIndex *index, const Dataset &ds, const DatasetConfig &conf, bool use_half_gt)
{
    std::string entry_file = ds.explore_entry_vertex_file();
    std::string explore_gt_file = ds.explore_groundtruth_file(use_half_gt);

    if (diskann::benchmark::file_exists(entry_file) && diskann::benchmark::file_exists(explore_gt_file))
    {
        log("\n--- Exploration Test (k=%u) ---\n", conf.explore_k);

        size_t entry_count = 0;
        auto entry_indices = load_ivecs_as_vectors(entry_file.c_str(), entry_count);

        size_t dim_gt = 0, n_gt = 0;
        auto gt_ptr = ivecs_read(explore_gt_file.c_str(), dim_gt, n_gt);

        std::vector<std::vector<uint32_t>> explore_gt_vec(n_gt);
        if (gt_ptr)
        {
            for (size_t i = 0; i < n_gt; ++i)
            {
                explore_gt_vec[i].assign(gt_ptr.get() + i * dim_gt, gt_ptr.get() + (i + 1) * dim_gt);
                std::sort(explore_gt_vec[i].begin(), explore_gt_vec[i].end());
            }
        }

        unsigned num_explore = 0, dim_explore = 0;
        float *explore_queries =
            load_fvecs((ds.files_dir() / ds.info().explore_query_file).string().c_str(), num_explore, dim_explore);
        size_t aligned_dim_explore = ROUND_UP(dim_explore, 8);

        auto typed_index = dynamic_cast<diskann::Index<float, uint32_t, uint32_t> *>(index);
        if (typed_index && explore_queries)
        {
            test_diskann_explore<float, uint32_t, uint32_t>(typed_index, explore_queries, num_explore, dim_explore,
                                                            aligned_dim_explore, explore_gt_vec, entry_indices,
                                                            conf.explore_k);
        }

        if (explore_queries)
            delete[] explore_queries;
    }
}

void run_test_suite(const Dataset &ds, const DatasetConfig &conf, bool force_test, bool only_test, uint32_t num_threads)
{
    std::string index_path = get_index_path(ds, conf);

    ensure_directory(ds.dataset_dir() / "diskann");
    std::string log_file = index_path + "_benchmark.log";

    if (!force_test && !only_test && diskann::benchmark::file_exists(log_file))
    {
        log("Log file %s already exists. Skipping.\n", log_file.c_str());
        return;
    }

    set_log_file(log_file, force_test || only_test);
    attach_cout_to_log();

    log("================================================================================\n");
    log("DiskANN Benchmark for %s\n", ds.name());
    log("================================================================================\n");

    try
    {
        if (!only_test && !diskann::benchmark::file_exists(index_path + "_pq_pivots.bin") &&
            !diskann::benchmark::file_exists(index_path + "_sample_data.bin") &&
            !diskann::benchmark::file_exists(index_path + ".data"))
        {
            run_create_index(index_path, ds, conf, num_threads);
        }
        else if (!only_test)
        {
            log("Index files already exist at %s. Skipping build.\n", index_path.c_str());
        }

        log("Loading query data...\n");
        auto query_data = ds.load_query();
        size_t query_num = ds.info().query_count;
        size_t query_dim = ds.info().dims;
        size_t query_aligned_dim = ROUND_UP(query_dim, 8);

        auto ground_truth = ds.load_groundtruth(conf.anns_k);

        log("Loading index for tests...\n");
        auto config = diskann::IndexConfigBuilder()
                          .with_metric(ds.info().metric)
                          .with_dimension(query_dim)
                          .with_max_points(ds.info().base_count)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .with_data_type("float")
                          .with_label_type("uint")
                          .is_dynamic_index(true)
                          .is_enable_tags(true)
                          .build();

        auto index_factory = diskann::IndexFactory(config);
        auto index = index_factory.create_instance();
        index->load(index_path.c_str(), num_threads, *(std::max_element(conf.Lvec.begin(), conf.Lvec.end())));

        log("----------------------------------------\n");
        log("Running ANNS Tests\n");
        log("----------------------------------------\n");

        if (ds.info().metric == diskann::FAST_L2)
        {
            log("Optimizing index layout for FAST_L2...\n");
            // Cast to exact Index to call optimize_layout if we can,
            // but IndexFactory returns AbstractIndex so we may need dynamic_cast
            auto typed_index = dynamic_cast<diskann::Index<float, uint32_t, uint32_t> *>(index.get());
            if (typed_index)
            {
                typed_index->optimize_index_layout();
            }
        }

        auto typed_index = dynamic_cast<diskann::Index<float, uint32_t, uint32_t> *>(index.get());
        if (typed_index)
        {
            test_diskann_anns<float, uint32_t, uint32_t>(typed_index, query_data.data, query_num, query_dim,
                                                         query_aligned_dim, ground_truth, conf.anns_k, conf.Lvec,
                                                         num_threads);

            run_explore_test(index.get(), ds, conf, false);
        }
        else
        {
            log("Failed to dynamic cast index for testing.\n");
        }
    }
    catch (const std::exception &e)
    {
        log("Exception during test: %s\n", e.what());
    }

    detach_cout_from_log();
    reset_log_to_console();

    // Now run the dynamic tests, appending to the single unified process
    run_dynamic_data_test(ds, conf, force_test, num_threads);
}

int main(int argc, char **argv)
{
    log("DiskANN Benchmark Suite\n");

#ifdef USE_AVX2
    log("Compiled with AVX2 support.\n");
#elif defined(__AVX__)
    log("Compiled with AVX support.\n");
#else
    log("Compiled without AVX/AVX2 support.\n");
#endif

    std::string data_root = DATA_PATH;
    DatasetName ds_name = DatasetName::AUDIO;
    bool only_test = false;
    bool force_test = false;
    uint32_t num_threads = omp_get_num_procs();

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--only-test" || arg == "-t")
        { // Added -t for consistency with original Boost.PO
            only_test = true;
        }
        else if (arg == "--force-test" || arg == "-f")
        { // Added -f
            force_test = true;
        }
        else if (arg == "-T" || arg == "--num_threads")
        {
            if (i + 1 < argc)
            {
                num_threads = std::stoi(argv[++i]);
            }
            else
            {
                std::cerr << "Error: --num_threads requires an argument.\n";
                return 1;
            }
        }
        else if (arg.find("--") == 0)
        {
            // Other flags ignored, or could add error for unknown flags
            std::cerr << "Warning: Unknown option " << arg << "\n";
        }
        else
        {
            // Must be dataset or path
            DatasetName params_ds = DatasetName::from_string(arg);
            if (params_ds.is_valid())
            {
                ds_name = params_ds;
            }
            else
            {
                // Assume it's the data path if not a valid dataset name
                data_root = arg;
            }
        }
    }

    log("Using %u threads for operations.\n", num_threads);

    if (!std::filesystem::exists(data_root))
    {
        std::cerr << "Data path does not exist: " << data_root << "\n";
        return 1;
    }

    std::vector<DatasetName> datasets_to_run;
    if (ds_name == DatasetName::ALL)
    {
        for (auto ds : DatasetName::all())
            datasets_to_run.push_back(ds);
    }
    else
    {
        datasets_to_run.push_back(ds_name);
    }

    for (const auto &ds_name_to_run : datasets_to_run)
    {
        Dataset dataset(ds_name_to_run, data_root);
        DatasetConfig conf = get_dataset_config(ds_name_to_run);
        run_test_suite(dataset, conf, force_test, only_test, num_threads);
    }

    return 0;
}
