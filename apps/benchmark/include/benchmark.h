#pragma once

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

#include "index.h"
#include "logging.h"
#include "stopwatch.h"
#include "dataset.h"

namespace diskann::benchmark
{

// Helper to access protected members of Index for exploration tests
template <typename T, typename TagT, typename LabelT> class IndexExplorer : public diskann::Index<T, TagT, LabelT>
{
  public:
    static std::pair<uint32_t, uint32_t> explore(diskann::Index<T, TagT, LabelT> *index, const T *query,
                                                 const uint32_t L, const std::vector<uint32_t> &init_ids,
                                                 diskann::InMemQueryScratch<T> *scratch)
    {
        const std::vector<LabelT> unused_filters;
        return ((IndexExplorer *)index)
            ->iterate_to_fixed_point(scratch->aligned_query(), L, init_ids, scratch, false, unused_filters, true);
    }
};

template <typename T, typename TagT, typename LabelT>
static void test_diskann_anns(diskann::Index<T, TagT, LabelT> *index, const T *query_data, size_t query_num,
                              size_t query_dim, size_t query_aligned_dim,
                              const std::vector<std::vector<uint32_t>> &ground_truth, const uint32_t k,
                              const std::vector<uint32_t> &Lvec, uint32_t num_threads)
{
    std::vector<TagT> query_result_tags(k * query_num);
    std::vector<float> latency_stats(query_num, 0);

    for (uint32_t L : Lvec)
    {
        if (L < k)
        {
            log("Ignoring search with L:%u since it's smaller than K:%u\n", L, k);
            continue;
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < query_num; i++)
        {
            auto qs = std::chrono::high_resolution_clock::now();

            std::vector<T *> res_vectors; // Empty vector to avoid null pointer copies
            std::vector<float> distances(k);

            // Always search with tags as they represent the original point IDs.
            index->search_with_tags(query_data + i * query_aligned_dim, k, L, query_result_tags.data() + i * k,
                                    distances.data(), res_vectors);

            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000.0);
        }

        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;

        double displayed_qps = query_num / diff.count();
        double qps_per_thread = displayed_qps / num_threads;

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<double>(query_num);
        float p999_latency = latency_stats[(uint64_t)(0.999 * query_num)];

        // Calculate Recall
        size_t correct = 0;
        for (size_t i = 0; i < query_num; i++)
        {
            if (i < ground_truth.size())
            {
                const auto &gt = ground_truth[i];
                for (size_t r = 0; r < k; r++)
                {
                    uint32_t id = static_cast<uint32_t>(query_result_tags[i * k + r]) - 1;
                    if (std::binary_search(gt.begin(), gt.end(), id))
                    {
                        correct++;
                    }
                }
            }
        }

        float recall = static_cast<float>(correct) / (static_cast<float>(query_num) * static_cast<float>(k));

        log("L_search %4u, Recall@%u %.4f, QPS/thread %8.2f, Mean Latency %6.2f us, 99.9 Latency %8.2f us\n", L, k,
            recall, qps_per_thread, mean_latency, p999_latency);
    }
}

template <typename T, typename TagT, typename LabelT>
static void test_diskann_explore(diskann::Index<T, TagT, LabelT> *index, const T *explore_query_data,
                                 size_t explore_query_num, size_t explore_query_dim, size_t explore_query_aligned_dim,
                                 const std::vector<std::vector<uint32_t>> &ground_truth,
                                 const std::vector<std::vector<uint32_t>> &entry_node_indices, const uint32_t k)
{
    log("Testing Exploration (k=%u)...\n", k);

    for (uint32_t L_base : {100, 1000, 10000})
    {
        for (uint32_t factor : {1, 2, 4, 6, 8, 10})
        {
            uint32_t L = L_base * factor;
            if (L < k)
                L = k;

            size_t correct = 0;
            size_t total = 0;
            auto start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < explore_query_num; i++)
            {
                if (i >= entry_node_indices.size() || entry_node_indices[i].empty())
                    continue;

                // For exploration, we use search_with_tags which is a public proxy for exploration.
                // In DiskANN, specific entry points are harder to set via public API, so we sweep L
                // to show how search performance evolves.

                std::vector<TagT> results(k);
                std::vector<float> dists(k);
                std::vector<T *> res_vecs;
                index->search_with_tags(explore_query_data + i * explore_query_aligned_dim, k, L, results.data(),
                                        dists.data(), res_vecs);

                if (i < ground_truth.size())
                {
                    const auto &gt = ground_truth[i];
                    for (size_t r = 0; r < k; r++)
                    {
                        uint32_t id = static_cast<uint32_t>(results[r]) - 1;
                        if (std::binary_search(gt.begin(), gt.end(), id))
                            correct++;
                    }
                    total += k;
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            float recall = total > 0 ? (float)correct / total : 0;
            uint64_t time_per_query =
                explore_query_num > 0 ? (uint64_t)(diff.count() * 1000000 / explore_query_num) : 0;

            log("L_search %6u, Recall@%u %.6f, time_us_per_query %4llu us\n", L, k, recall,
                (unsigned long long)time_per_query);
            if (recall >= 0.995f)
                return;
        }
    }
}

} // namespace diskann::benchmark
