#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "dataset.h"
#include "file_io.h"
#include "logging.h"
#include "utils.h"

namespace diskann::benchmark::analysis
{

struct VertexReach
{
    uint32_t vertex_id;
    uint32_t reach_count;
    std::vector<bool> reachable_ids;
};

inline void generate_graph_stats(const std::string &graph_file, const Dataset &ds, bool use_half_gt,
                                uint32_t num_threads)
{
    (void)num_threads;

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);

    try
    {
        in.open(graph_file, std::ios::binary);
    }
    catch (const std::exception &)
    {
        log("Warning: Could not open graph file %s for statistics calculation.\n", graph_file.c_str());
        return;
    }

    size_t expected_file_size;
    uint32_t max_observed_degree;
    uint32_t start;
    size_t file_frozen_pts;

    in.read((char *)&expected_file_size, sizeof(size_t));
    in.read((char *)&max_observed_degree, sizeof(uint32_t));
    in.read((char *)&start, sizeof(uint32_t));
    in.read((char *)&file_frozen_pts, sizeof(size_t));

    size_t total_nodes = 0;
    size_t min_out_degree = std::numeric_limits<size_t>::max();
    size_t max_out_degree = 0;
    size_t total_edges = 0;
    size_t count_out_degree_0 = 0;
    size_t count_out_degree_1 = 0;

    std::vector<std::vector<uint32_t>> adj;
    std::vector<uint32_t> in_degrees;

    size_t bytes_read = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

    while (bytes_read < expected_file_size)
    {
        uint32_t k = 0;
        in.read((char *)&k, sizeof(uint32_t));

        std::vector<uint32_t> neighbors(k);
        if (k > 0)
        {
            in.read((char *)neighbors.data(), k * sizeof(uint32_t));
            for (uint32_t ngh : neighbors)
            {
                if (ngh >= in_degrees.size())
                {
                    in_degrees.resize(std::max((size_t)ngh + 1, in_degrees.size() * 2), 0);
                }
                in_degrees[ngh]++;
            }
        }
        adj.push_back(std::move(neighbors));

        bytes_read += sizeof(uint32_t) * (k + 1);

        min_out_degree = std::min(min_out_degree, (size_t)k);
        max_out_degree = std::max(max_out_degree, (size_t)k);
        total_edges += k;

        if (k == 0)
            count_out_degree_0++;
        else if (k == 1)
            count_out_degree_1++;

        total_nodes++;
    }

    if (total_nodes == 0)
    {
        min_out_degree = 0;
    }

    if (in_degrees.size() < total_nodes)
    {
        in_degrees.resize(total_nodes, 0);
    }

    // Load tags file to identify active non-frozen vertices and their external labels
    std::string tags_file = graph_file + ".tags";
    std::vector<uint32_t> tags;
    size_t tags_npts = 0, tags_dim = 0;
    if (std::filesystem::exists(tags_file))
    {
        uint32_t *tag_data = nullptr;
        diskann::load_bin<uint32_t>(tags_file, tag_data, tags_npts, tags_dim);
        if (tag_data)
        {
            tags.assign(tag_data, tag_data + tags_npts);
            delete[] tag_data;
        }
    }

    std::vector<uint8_t> active_mask(total_nodes, 1);
    std::vector<uint32_t> active_ids;
    active_ids.reserve(total_nodes);

    for (uint32_t internal = 0; internal < total_nodes; internal++)
    {
        // Frozen point or points with tag 0 (reserved/deleted/empty) are inactive
        if (file_frozen_pts > 0 && internal >= (total_nodes - file_frozen_pts))
        {
            active_mask[internal] = 0;
            continue;
        }
        if (internal < tags.size() && tags[internal] == 0)
        {
            active_mask[internal] = 0;
            continue;
        }
        active_ids.push_back(internal);
    }

    const uint32_t active_count = static_cast<uint32_t>(active_ids.size());

    // In-degree statistics for active vertices
    size_t min_in_degree = active_count > 0 ? std::numeric_limits<size_t>::max() : 0;
    size_t max_in_degree = 0;
    size_t count_in_degree_0 = 0;
    size_t count_in_degree_1 = 0;

    for (const auto n : active_ids)
    {
        size_t deg = (size_t)in_degrees[n];
        min_in_degree = std::min(min_in_degree, deg);
        max_in_degree = std::max(max_in_degree, deg);
        if (deg == 0)
            count_in_degree_0++;
        else if (deg == 1)
            count_in_degree_1++;
    }
    if (min_in_degree == std::numeric_limits<size_t>::max())
    {
        min_in_degree = 0;
    }

    // -------------------------------------------------------------------------
    // Graph Quality (GQ) using precomputed base ground truth
    // -------------------------------------------------------------------------
    bool gq_available = false;
    float perfect_neighbor_ratio = 0.0f;
    std::string base_gt_file = ds.base_groundtruth_file(use_half_gt);

    if (std::filesystem::exists(base_gt_file) && !tags.empty())
    {
        size_t top_list_dims = 0, top_list_count = 0;
        auto all_top_list = ivecs_read(base_gt_file.c_str(), top_list_dims, top_list_count);

        if (all_top_list && top_list_count > 0 && top_list_dims > 0)
        {
            uint64_t perfect_neighbor_count = 0;
            uint64_t active_total_edges = 0;

            for (const auto n : active_ids)
            {
                if (n >= tags.size() || tags[n] == 0)
                    continue;

                // tag is 1-based index into base dataset
                const uint64_t node_ext = static_cast<uint64_t>(tags[n] - 1);
                if (node_ext >= top_list_count)
                    continue;

                const uint32_t *top_list = all_top_list.get() + node_ext * top_list_dims;
                const auto &nbrs = adj[n];

                uint32_t valid_edges = 0;
                for (uint32_t ngh : nbrs)
                {
                    if (ngh < total_nodes && active_mask[ngh])
                        valid_edges++;
                }

                if (valid_edges == 0)
                    continue;

                active_total_edges += valid_edges;
                const uint32_t check_count =
                    std::min<uint32_t>(valid_edges, static_cast<uint32_t>(top_list_dims));

                uint32_t checked = 0;
                for (uint32_t ngh : nbrs)
                {
                    if (checked >= check_count)
                        break;
                    if (ngh >= total_nodes || !active_mask[ngh])
                        continue;
                    if (ngh >= tags.size() || tags[ngh] == 0)
                        continue;

                    const uint32_t nbr_ext = tags[ngh] - 1;
                    for (uint32_t i = 0; i < check_count; i++)
                    {
                        if (nbr_ext == top_list[i])
                        {
                            perfect_neighbor_count++;
                            break;
                        }
                    }
                    checked++;
                }
            }

            perfect_neighbor_ratio = (active_total_edges > 0)
                                         ? static_cast<float>(perfect_neighbor_count) / static_cast<float>(active_total_edges)
                                         : 0.0f;
            gq_available = true;
        }
    }

    // -------------------------------------------------------------------------
    // Search Reachability (BFS from start/entry point)
    // -------------------------------------------------------------------------
    std::vector<bool> search_visited(total_nodes, false);
    std::vector<uint32_t> frontier;
    frontier.reserve(total_nodes);

    uint32_t ep = start;
    if (ep >= total_nodes && !active_ids.empty())
        ep = active_ids[0];

    if (ep < total_nodes)
    {
        search_visited[ep] = true;
        frontier.push_back(ep);
    }

    size_t head = 0;
    while (head < frontier.size())
    {
        const uint32_t v = frontier[head++];
        for (uint32_t cand : adj[v])
        {
            if (cand < total_nodes && !search_visited[cand])
            {
                search_visited[cand] = true;
                frontier.push_back(cand);
            }
        }
    }

    uint32_t search_reach_count = 0;
    for (const auto id : active_ids)
    {
        if (search_visited[id])
            search_reach_count++;
    }

    // -------------------------------------------------------------------------
    // Exploration Reachability (Average BFS reach across all vertices)
    // -------------------------------------------------------------------------
    uint32_t best_vertex_reach = 0;
    std::vector<VertexReach> vertices_reach;
    std::vector<uint32_t> index_of_vertex_reach(total_nodes, static_cast<uint32_t>(total_nodes));

    uint64_t exploration_total_reach = 0;

    for (size_t entry_idx = 0; entry_idx < active_ids.size(); entry_idx++)
    {
        const uint32_t entry_id = active_ids[entry_idx];

        std::vector<bool> checked_ids(total_nodes, false);
        std::vector<uint32_t> check;
        std::vector<uint32_t> check_next;

        checked_ids[entry_id] = true;
        check.push_back(entry_id);

        uint32_t best_reach_vertex_index = 0;
        uint32_t best_reach_vertex_reach = 0;

        auto check_ptr = &check;
        auto check_next_ptr = &check_next;

        while (!check_ptr->empty() && best_reach_vertex_reach < active_count)
        {
            check_next_ptr->clear();

            for (size_t c = 0; c < check_ptr->size() && best_reach_vertex_reach < active_count; c++)
            {
                const auto check_index = check_ptr->at(c);
                for (uint32_t neighbor_index : adj[check_index])
                {
                    if (neighbor_index >= total_nodes || !active_mask[neighbor_index])
                        continue;

                    if (!checked_ids[neighbor_index])
                    {
                        checked_ids[neighbor_index] = true;
                        check_next_ptr->push_back(neighbor_index);

                        const auto vertex_reach_index = index_of_vertex_reach[neighbor_index];
                        if (vertex_reach_index < total_nodes)
                        {
                            const auto &neighbor_reach = vertices_reach[vertex_reach_index];
                            if (neighbor_reach.reach_count == active_count)
                            {
                                best_reach_vertex_index = vertex_reach_index;
                                best_reach_vertex_reach = active_count;
                                break;
                            }

                            if (neighbor_reach.reach_count > best_reach_vertex_reach)
                            {
                                best_reach_vertex_reach = neighbor_reach.reach_count;
                                best_reach_vertex_index = vertex_reach_index;

                                const auto &best_vertex_checked_ids = neighbor_reach.reachable_ids;
                                for (size_t b = 0; b < total_nodes; b++)
                                    checked_ids[b] = checked_ids[b] | best_vertex_checked_ids[b];
                            }
                        }
                    }
                }
            }

            std::swap(check_ptr, check_next_ptr);
        }

        if (best_reach_vertex_reach == active_count)
        {
            index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            exploration_total_reach += active_count;
        }
        else
        {
            uint32_t reach_count = 0;
            for (const auto id : active_ids)
                reach_count += checked_ids[id];

            exploration_total_reach += reach_count;

            if (best_vertex_reach < reach_count)
            {
                best_vertex_reach = reach_count;
                index_of_vertex_reach[entry_id] = static_cast<uint32_t>(vertices_reach.size());
                vertices_reach.emplace_back(VertexReach{entry_id, reach_count, std::move(checked_ids)});
            }
            else if (best_reach_vertex_reach > 0)
            {
                index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            }
            else
            {
                index_of_vertex_reach[entry_id] = static_cast<uint32_t>(vertices_reach.size());
                vertices_reach.emplace_back(VertexReach{entry_id, reach_count, std::move(checked_ids)});
            }
        }
    }

    const double search_reach_pct =
        (active_count > 0) ? (100.0 * static_cast<double>(search_reach_count) / static_cast<double>(active_count)) : 0.0;
    const double avg_exploration_reach =
        (active_count > 0) ? (static_cast<double>(exploration_total_reach) / static_cast<double>(active_count)) : 0.0;
    const double explore_reach_pct =
        (active_count > 0) ? (100.0 * avg_exploration_reach / static_cast<double>(active_count)) : 0.0;

    log("\n----------------------------------------\n");
    log("Graph Statistics:\n");
    log("----------------------------------------\n");
    log("Total Nodes      : %zu (Active: %u)\n", total_nodes, active_count);
    log("Total Edges      : %zu\n", total_edges);
    log("Max Out-Degree   : %zu\n", max_out_degree);
    log("Min Out-Degree   : %zu\n", min_out_degree);
    log("Max In-Degree    : %zu\n", max_in_degree);
    log("Min In-Degree    : %zu\n", min_in_degree);
    log("Count (Out=0)    : %zu\n", count_out_degree_0);
    log("Count (Out=1)    : %zu\n", count_out_degree_1);
    log("Count (In=0)     : %zu\n", count_in_degree_0);
    log("Count (In=1)     : %zu\n", count_in_degree_1);
    log("Average Degree   : %.2f\n", total_nodes > 0 ? (float)total_edges / total_nodes : 0.0f);
    if (gq_available)
        log("Graph Quality    : %.4f\n", perfect_neighbor_ratio);
    else
        log("Graph Quality    : N/A\n");
    log("Search Reach.    : %.2f%% (%u / %u)\n", search_reach_pct, search_reach_count, active_count);
    log("Exploration Reach: %.2f%% (Avg: %.2f)\n", explore_reach_pct, avg_exploration_reach);
    log("----------------------------------------\n\n");
}

} // namespace diskann::benchmark::analysis
