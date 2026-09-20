#pragma once

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "dataset.h"
#include "file_io.h"
#include "hnswlib.h"
#include "logging.h"
#include "stopwatch.h"

// Helper to read ivecs for stats
namespace hnswlib::benchmark::stats {

using tableint = hnswlib::tableint;

static uint32_t compute_search_reachability(hnswlib::HierarchicalNSW<float>* graph,
                                            const std::vector<tableint>& active_ids,
                                            const std::vector<uint8_t>& active_mask) {
    const auto full_count = (tableint)graph->cur_element_count;
    const auto graph_size = (uint32_t)active_ids.size();
    auto stopw = StopW();

    std::vector<bool> visited(full_count, false);
    std::vector<tableint> frontier;
    frontier.reserve(full_count);

    // Start from the graph's entry point
    if (!active_ids.empty()) {
        tableint ep = graph->enterpoint_node_;
        // Use entry point even if inactive (it's the graph entry), fallback only if OOB
        if (ep >= full_count) ep = active_ids[0];

        visited[ep] = true;
        frontier.push_back(ep);
    }

    size_t head = 0;
    while (head < frontier.size()) {
        const tableint v = frontier[head++];

        // Follow edges on level 0
        unsigned int* data = (unsigned int*)graph->get_linklist0(v);
        const int size = graph->getListCount(data);
        const tableint* neighbor_indices = (tableint*)(data + 1);

        for (int i = 0; i < size; i++) {
            const tableint cand = neighbor_indices[i];
            if (cand < full_count && !visited[cand]) {
                visited[cand] = true;
                frontier.push_back(cand);
            }
        }
    }

    uint32_t reachable_count = 0;
    for (const auto id : active_ids) {
        if (visited[id]) reachable_count++;
    }

    log("Seed Reachability is {} out of {} after {}s\n", reachable_count, graph_size, stopw.getElapsedTimeMicro() / 1000000);
    return reachable_count;
}

struct VertexReach {
    uint32_t vertex_id;
    uint32_t reach_count;
    std::vector<bool> reachable_ids;
};

static float compute_exploration_reach(hnswlib::HierarchicalNSW<float>* graph,
                                       const std::vector<tableint>& active_ids,
                                       const std::vector<uint8_t>& active_mask) {
    const auto full_count = static_cast<tableint>(graph->cur_element_count);
    const auto graph_size = static_cast<tableint>(active_ids.size());
    auto stopw = StopW();

    uint32_t best_vertex_reach = 0;
    auto vertices_reach = std::vector<VertexReach>();
    auto index_of_vertex_reach = std::vector<uint32_t>(full_count);
    std::fill(index_of_vertex_reach.begin(), index_of_vertex_reach.end(), static_cast<uint32_t>(full_count));

    uint64_t counter = 0;
    uint64_t avg_reach = 0;

    // NOTE: This is expensive (potentially very slow for large graphs), kept to match the legacy tool output.
    for (size_t entry_idx = 0; entry_idx < active_ids.size(); entry_idx++) {
        const tableint entry_id = active_ids[entry_idx];

        auto checked_ids = std::vector<bool>(full_count);
        auto check = std::vector<tableint>();
        auto check_next = std::vector<tableint>();

        checked_ids[entry_id] = true;
        check.emplace_back(entry_id);

        uint32_t best_reach_vertex_index = 0;
        uint32_t best_reach_vertex_reach = 0;

        auto check_ptr = &check;
        auto check_next_ptr = &check_next;
        while (check_ptr->size() > 0 && best_reach_vertex_reach < graph_size) {
            check_next_ptr->clear();

            for (size_t c = 0; c < check_ptr->size() && best_reach_vertex_reach < graph_size; c++) {
                const auto check_index = check_ptr->at(c);
                unsigned int* data = (unsigned int*)graph->get_linklist0(check_index);
                const int size = graph->getListCount(data);

                if (size == 0) log("zero out-degree for vertex {}\n", check_index);

                const tableint* neighbor_indices = (tableint*)(data + 1);
                for (int n = 0; n < size; n++) {
                    const tableint neighbor_index = neighbor_indices[n];
                    if (neighbor_index >= full_count) continue;

                    if (!active_mask[neighbor_index]) continue;

                    if (checked_ids[neighbor_index] == false) {
                        checked_ids[neighbor_index] = true;
                        check_next_ptr->emplace_back(neighbor_index);

                        const auto vertex_reach_index = index_of_vertex_reach[neighbor_index];
                        if (vertex_reach_index < full_count) {
                            const auto& neighbor_reach = vertices_reach[vertex_reach_index];
                            if (neighbor_reach.reach_count == graph_size) {
                                best_reach_vertex_index = vertex_reach_index;
                                best_reach_vertex_reach = graph_size;
                                break;
                            }

                            if (neighbor_reach.reach_count > best_reach_vertex_reach) {
                                best_reach_vertex_reach = neighbor_reach.reach_count;
                                best_reach_vertex_index = vertex_reach_index;

                                const auto& best_vertex_checked_ids = neighbor_reach.reachable_ids;
                                for (size_t b = 0; b < full_count; b++) checked_ids[b] = checked_ids[b] | best_vertex_checked_ids[b];
                            }
                        }
                    }
                }
            }

            auto buffer = check_ptr;
            check_ptr = check_next_ptr;
            check_next_ptr = buffer;
        }

        if (best_reach_vertex_reach == graph_size) {
            index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            avg_reach += graph_size;

        } else {
            uint32_t reach_count = 0;
            for (const auto id : active_ids) reach_count += checked_ids[id];
            avg_reach += reach_count;

            if (best_vertex_reach < reach_count) {
                best_vertex_reach = reach_count;
                index_of_vertex_reach[entry_id] = (uint32_t)vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
            } else if (best_reach_vertex_reach > 0) {
                index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            } else {
                index_of_vertex_reach[entry_id] = (uint32_t)vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
            }
        }

        counter++;
    }

    log("Avg reach is {:.2f} after checking {:7d} of {:7d} vertices after {:4d}s\n",
        (static_cast<float>(avg_reach)) / static_cast<float>(counter),
        counter,
        graph_size,
        stopw.getElapsedTimeMicro() / 1000000);
    return (graph_size > 0) ? (static_cast<float>(avg_reach) / static_cast<float>(graph_size)) : 0.0f;
}

/**
 * Compute statistics about the graph quality and reachability.
 * Output is modeled after the legacy `hnswlib_graph_stats.cpp` logs.
 */
static void compute_stats(hnswlib::HierarchicalNSW<float>* graph, const char* top_list_file, uint32_t feature_dims) {
    log("\n--- Graph Analysis ---\n");

    const auto full_count = static_cast<uint32_t>(graph->cur_element_count);
    std::vector<uint8_t> active_mask(full_count, 1);
    std::vector<tableint> active_ids;
    active_ids.reserve(full_count);
    for (uint32_t internal = 0; internal < full_count; internal++) {
        if (graph->isMarkedDeleted(internal)) {
            active_mask[internal] = 0;
            continue;
        }
        active_ids.push_back(static_cast<tableint>(internal));
    }

    const auto graph_size = static_cast<uint32_t>(active_ids.size());
    const auto max_edges_per_node = graph->maxM0_;

    if (graph_size != full_count) {
        log("Active vertices: {} (cur_element_count={})\n", graph_size, full_count);
    }

    log("Computing graph quality...\n");

    uint64_t total_edges = 0;
    uint16_t min_out = static_cast<uint16_t>(max_edges_per_node);
    uint16_t max_out = 0;

    auto in_degree_count = std::vector<uint32_t>(full_count);

    // Always compute degree stats from the graph itself.
    min_out = std::numeric_limits<uint16_t>::max();
    for (const auto n : active_ids) {
        auto linklist_data = graph->get_linklist_at_level(n, 0);
        const auto raw_edges_per_node = static_cast<uint16_t>(graph->getListCount(linklist_data));
        auto neighbor_indices = (tableint*)(linklist_data + 1);

        uint16_t edges_per_node = 0;
        for (uint32_t e = 0; e < raw_edges_per_node; e++) {
            const tableint neighbor_index = neighbor_indices[e];
            if (static_cast<uint32_t>(neighbor_index) >= full_count) {
                continue;
            }
            if (!active_mask[static_cast<uint32_t>(neighbor_index)]) {
                continue;
            }
            edges_per_node++;
        }

        if (edges_per_node < min_out) min_out = edges_per_node;
        if (max_out < edges_per_node) max_out = edges_per_node;

        total_edges += edges_per_node;
        for (uint32_t e = 0; e < raw_edges_per_node; e++) {
            const tableint neighbor_index = neighbor_indices[e];
            if (static_cast<uint32_t>(neighbor_index) >= full_count) {
                continue;
            }
            if (!active_mask[static_cast<uint32_t>(neighbor_index)]) {
                continue;
            }
            in_degree_count[static_cast<uint32_t>(neighbor_index)]++;
        }
    }

    if (min_out == std::numeric_limits<uint16_t>::max()) {
        min_out = 0;
    }

    uint32_t min_in = 0;
    uint32_t max_in = 0;
    uint32_t source_vertices = 0;
    if (graph_size > 0) {
        min_in = std::numeric_limits<uint32_t>::max();
    }
    for (const auto n : active_ids) {
        const uint32_t in_deg = in_degree_count[static_cast<uint32_t>(n)];
        if (in_deg < min_in) min_in = in_deg;
        if (max_in < in_deg) max_in = in_deg;
        if (in_deg == 0) source_vertices++;
    }

    if (min_in == std::numeric_limits<uint32_t>::max()) {
        min_in = 0;
    }

    // Graph quality (GQ) is optional: requires a matching TopList file.
    bool gq_available = false;
    float perfect_neighbor_ratio = 0.0f;
    {
        size_t top_list_dims = 0;
        size_t top_list_count = 0;
        auto all_top_list = ivecs_read(top_list_file, top_list_dims, top_list_count);

        if (!all_top_list) {
            log("Skipping graph quality: could not load TopList file {}\n", top_list_file);
        } else if (top_list_count < 1) {
            log("Skipping graph quality: TopList element count mismatch: {} vs {}\n", top_list_count, graph_size);
        } else if (top_list_dims < 1) {
            log("Skipping graph quality: TopList has invalid k={}\n", top_list_dims);
        } else {
            uint64_t max_label = 0;
            for (const auto internal : active_ids) {
                const auto lbl = static_cast<uint64_t>(graph->getExternalLabel(internal));
                if (lbl > max_label) max_label = lbl;
            }

            if (top_list_count <= max_label) {
                log("Skipping graph quality: TopList element count mismatch: {} vs {}\n", top_list_count, (max_label + 1));
            } else {
                uint64_t perfect_neighbor_count = 0;
                for (const auto n : active_ids) {
                    auto linklist_data = graph->get_linklist_at_level(n, 0);
                    const auto raw_edges_per_node = static_cast<uint16_t>(graph->getListCount(linklist_data));
                    auto neighbor_indices = (tableint*)(linklist_data + 1);

                    const auto node_label = static_cast<uint64_t>(graph->getExternalLabel(n));
                    const auto top_list = all_top_list.get() + node_label * top_list_dims;

                    uint32_t edges_per_node = 0;
                    for (uint32_t e = 0; e < raw_edges_per_node; e++) {
                        const tableint neighbor_internal = neighbor_indices[e];
                        if (static_cast<uint32_t>(neighbor_internal) >= full_count) continue;
                        if (!active_mask[static_cast<uint32_t>(neighbor_internal)]) continue;
                        edges_per_node++;
                    }

                    const uint32_t check_count =
                        std::min<uint32_t>(static_cast<uint32_t>(edges_per_node), static_cast<uint32_t>(top_list_dims));

                    uint32_t checked_edges = 0;
                    for (uint32_t e = 0; e < raw_edges_per_node && checked_edges < check_count; e++) {
                        const tableint neighbor_internal = neighbor_indices[e];
                        if (static_cast<uint32_t>(neighbor_internal) >= full_count) continue;
                        if (!active_mask[static_cast<uint32_t>(neighbor_internal)]) continue;

                        const auto neighbor_label = static_cast<uint32_t>(graph->getExternalLabel(neighbor_internal));
                        for (uint32_t i = 0; i < check_count; i++) {
                            if (neighbor_label == top_list[i]) {
                                perfect_neighbor_count++;
                                break;
                            }
                        }
                        checked_edges++;
                    }
                }

                perfect_neighbor_ratio =
                    (total_edges > 0) ? static_cast<float>(perfect_neighbor_count) / static_cast<float>(total_edges) : 0.0f;
                gq_available = true;
            }
        }
    }

    log("Computing search reachability...\n");
    const auto reachability_count = compute_search_reachability(graph, active_ids, active_mask);

    log("Computing exploration reachability...\n");
    const auto avg_reach = compute_exploration_reach(graph, active_ids, active_mask);

    const double out_avg = (graph_size > 0) ? static_cast<double>(total_edges) / static_cast<double>(graph_size) : 0.0;
    const double in_avg = out_avg;
    const double search_reach_pct =
        (graph_size > 0) ? (100.0 * static_cast<double>(reachability_count) / static_cast<double>(graph_size)) : 0.0;
    const double explore_reach_pct = (graph_size > 0) ? (100.0 * static_cast<double>(avg_reach) / static_cast<double>(graph_size)) : 0.0;

    log("Graph Statistics:\n");
    log("  Vertices: {}\n", graph_size);
    log("  Total edges: {}\n", total_edges);
    log("  Feature dimensions: {}\n", feature_dims);
    log("  Avg Edges per vertex: {:.0f}\n", out_avg);
    log("  Out-degree: avg={:.2f}, min={}, max={}\n", out_avg, min_out, max_out);
    log("  In-degree:  avg={:.2f}, min={}, max={}, source_vertices={}\n", in_avg, min_in, max_in, source_vertices);
    if (gq_available) {
        log("  Graph Quality (GQ): {:.4f}\n", perfect_neighbor_ratio);
    } else {
        log("  Graph Quality (GQ): N/A\n");
    }
    log("  Search Reachability: {:.2f}%\n", search_reach_pct);
    log("  Exploration Reachability: {:.2f}%\n", explore_reach_pct);
}

}  // namespace hnswlib::benchmark::stats
