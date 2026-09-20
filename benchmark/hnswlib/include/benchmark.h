#pragma once

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "dataset.h"
#include "file_io.h"
#include "hnswlib.h"
#include "logging.h"
#include "stopwatch.h"

namespace hnswlib::benchmark {

namespace detail {
template <typename Index, typename = void>
struct has_tryGetInternalId : std::false_type {};

template <typename Index>
struct has_tryGetInternalId<
    Index,
    std::void_t<decltype(std::declval<Index*>()->tryGetInternalId(std::declval<hnswlib::labeltype>(), std::declval<hnswlib::tableint&>()))>>
    : std::true_type {};
}  // namespace detail

template <typename Index>
static float test_approx_anns(Index* graph,
                              const VectorRepository& query_repository,
                              const std::vector<std::vector<uint32_t>>& ground_truth,
                              const int ef,
                              const uint32_t k,
                              const uint32_t test_size,
                              const uint32_t threads) {
    graph->setEf(ef);

    auto corrects = std::vector<float>(threads);
    parallel_for(0, test_size, threads, [&](size_t i, size_t thread_id) {
        auto query = query_repository.getFeature(uint32_t(i));
        // HNSW searchKnn returns priority queue of <dist, label> (max heap)
        auto result_queue = graph->searchKnn(query, k);

        uint32_t correct = 0;
        const auto& gt = ground_truth[i];

        while (!result_queue.empty()) {
            const auto item = result_queue.top();
            if (std::binary_search(gt.begin(), gt.end(), item.second)) correct++;
            result_queue.pop();
        }

        corrects[thread_id] += correct;
    });

    float total_correct = 0;
    for (size_t i = 0; i < threads; i++) total_correct += corrects[i];
    return total_correct / (test_size * k);
}

template <typename Index>
static void test_graph_anns(Index* graph,
                            const VectorRepository& query_repository,
                            const std::vector<std::vector<uint32_t>>& ground_truth,
                            const uint32_t repeat,
                            const uint32_t threads,
                            const uint32_t k,
                            const std::vector<float>& ef_parameter) {
    // sort ef_parameter
    std::vector<float> ef_parameter_sorted = ef_parameter;
    std::sort(ef_parameter_sorted.begin(), ef_parameter_sorted.end());
    log("Compute TOP{} for ef {}\n", k, fmt::join(ef_parameter_sorted, ", "));

    const auto test_size = uint32_t(query_repository.size());
    for (float ef_f : ef_parameter_sorted) {
        int ef = (int)ef_f;

        graph->metric_distance_computations = 0;
        graph->metric_hops = 0;

        StopW stopw = StopW();
        float recall = 0;
        for (size_t i = 0; i < repeat; i++) recall = test_approx_anns(graph, query_repository, ground_truth, ef, k, test_size, threads);
        uint64_t search_time_us = stopw.getElapsedTimeMicro();
        uint64_t time_us_per_query = (search_time_us / test_size) / repeat;

        const double denom = static_cast<double>(test_size) * static_cast<double>(repeat);
        const double distance_comp_per_query =
            (denom > 0.0) ? static_cast<double>(graph->metric_distance_computations.load()) / denom : 0.0;
        const double hops_per_query = (denom > 0.0) ? static_cast<double>(graph->metric_hops.load()) / denom : 0.0;

        log("ef {:3} \t recall {:.5f} \t time_us_per_query {:6}us, avg distance computations {}, avg hops {}\n",
            ef,
            recall,
            time_us_per_query,
            distance_comp_per_query,
            hops_per_query);
        if (recall > 0.995) break;
    }
}

template <typename Index>
static float test_approx_explore(Index* graph,
                                 const std::vector<std::vector<uint32_t>>& ground_truth,
                                 const std::vector<std::vector<uint32_t>>& entry_node_indices,
                                 const uint32_t k,
                                 const uint32_t max_distance_count) {
    size_t correct = 0;
    size_t total = 0;

    // Serial execution in original explore code, keeping it simple
    for (size_t i = 0; i < ground_truth.size(); i++) {
        if (entry_node_indices[i].empty()) continue;
        const auto entry_node_label = static_cast<hnswlib::labeltype>(entry_node_indices[i][0]);

        // explore() expects an internal id. In most benchmarks label==internalId,
        // but dynamic scenarios can break that assumption.
        tableint entry_node = static_cast<tableint>(entry_node_label);
        if constexpr (detail::has_tryGetInternalId<Index>::value) {
            tableint internal = 0;
            if (!graph->tryGetInternalId(entry_node_label, internal)) {
                continue;
            }
            entry_node = internal;
        }

        const auto& gt = ground_truth[i];

        // Ensure Index has explore method
        auto result_queue = graph->explore(entry_node, k, max_distance_count);

        total += gt.size();
        while (result_queue.empty() == false) {
            if (std::binary_search(gt.begin(), gt.end(), result_queue.top().second)) correct++;
            result_queue.pop();
        }
    }

    return total > 0 ? 1.0f * correct / total : 0;
}

template <typename Index>
static void test_explore(Index* graph,
                         const std::vector<std::vector<uint32_t>>& ground_truth,
                         const std::vector<std::vector<uint32_t>>& entry_node_indices,
                         const uint32_t k) {
    log("Testing Exploration (k={})...\n", k);

    // try different k values
    uint32_t k_factor = 100;
    for (uint32_t f = 0; f <= 3; f++, k_factor *= 10) {
        for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {
            const auto max_distance_count = ((f == 0) ? (k + k_factor * (i - 1)) : (k_factor * i));

            graph->setEf(k);
            graph->metric_distance_computations = 0;
            graph->metric_hops = 0;

            auto stopw = StopW();
            auto recall = test_approx_explore(graph, ground_truth, entry_node_indices, k, max_distance_count);
            auto time_us_per_query = stopw.getElapsedTimeMicro() / ground_truth.size();

            const float denom = 1.0f * static_cast<float>(ground_truth.size());
            const float distance_comp_per_query =
                (denom > 0.0f) ? static_cast<float>(graph->metric_distance_computations.load()) / denom : 0.0f;
            const float hops_per_query = (denom > 0.0f) ? static_cast<float>(graph->metric_hops.load()) / denom : 0.0f;

            log("max_distance_count {:6d} \t recall {:.6f} \t time_us_per_query {:4d}us, avg distance computations {:8.2f}, avg hops "
                "{:6.2f}\n",
                max_distance_count,
                recall,
                time_us_per_query,
                distance_comp_per_query,
                hops_per_query);
            if (recall >= 0.995f) return;
        }
    }
}

}  // namespace hnswlib::benchmark
