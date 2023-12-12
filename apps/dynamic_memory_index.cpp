// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>
#include <filesystem>
#include <numeric>
#include <time.h>
#include <timer.h>

#include "index.h"
#include "utils.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

// load_aligned_bin modified to read pieces of the file, but using ifstream
// instead of cached_ifstream.
template <typename T>
inline void load_aligned_bin_part(const std::string &bin_file, T *data, size_t offset_points, size_t points_to_read)
{
    diskann::Timer timer;
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();
    reader.seekg(0, std::ios::beg);

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    size_t npts = (uint32_t)npts_i32;
    size_t dim = (uint32_t)dim_i32;

    size_t expected_actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size)
    {
        std::stringstream stream;
        stream << "Error. File size mismatch. Actual size is " << actual_file_size << " while expected size is  "
               << expected_actual_file_size << " npts = " << npts << " dim = " << dim << " size of <T>= " << sizeof(T)
               << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (offset_points + points_to_read > npts)
    {
        std::stringstream stream;
        stream << "Error. Not enough points in file. Requested " << offset_points << "  offset and " << points_to_read
               << " points, but have only " << npts << " points" << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    reader.seekg(2 * sizeof(uint32_t) + offset_points * dim * sizeof(T));

    const size_t rounded_dim = ROUND_UP(dim, 8);

    for (size_t i = 0; i < points_to_read; i++)
    {
        reader.read((char *)(data + i * rounded_dim), dim * sizeof(T));
        memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    reader.close();

    const double elapsedSeconds = timer.elapsed() / 1000000.0;
    std::cout << "Read " << points_to_read << " points using non-cached reads in " << elapsedSeconds << std::endl;
}

int main(int argc, char **argv)
{
    //const auto data_dir = std::filesystem::path("e:/Data/Feature/SIFT1M/DiskANN/");
    const auto data_dir = std::filesystem::path("e:/Data/Feature/GloVe/DiskANN/");

    std::string data_type = "float";
    std::string label_type = "uint";
    //std::string data_path = (data_dir / "sift_base.fbin").string();
    std::string data_path = (data_dir / "glove-100_base.fbin").string();
    //std::string index_path_prefix = (data_dir / "R64_L75_A1.2_add500k_add2_remove2_until500k_10pConsolidate.da").string();
    std::string index_path_prefix = (data_dir / "R64_L75_A1.2_add1183k_remove591k_10pConsolidat.da").string();

    uint32_t num_threads = 1;
    uint32_t R = 64;
    uint32_t L = 75;
    float alpha = 1.2;

    diskann::Metric metric = diskann::Metric::L2;

    try
    {
        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;

        size_t data_num, data_dim;
        diskann::get_bin_metadata(data_path, data_num, data_dim);
        size_t aligned_dim = ROUND_UP(data_dim, 8);

        float *data = nullptr;
        diskann::alloc_aligned((void **)&data, data_num * aligned_dim * sizeof(float), 8 * sizeof(float));
        load_aligned_bin_part(data_path, data, 0, data_num);
        diskann::cout << "Data " << data[10] << ", " << data[data_num - 10] << std::endl;

        std::vector<uint32_t> tags(data_num);
        std::iota(tags.begin(), tags.end(), 1); // tag 0 is reserved for hidden points
        diskann::cout << "Tags " << tags[0] << ", " << tags[data_num - 1] << std::endl;

        auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_filter_list_size(0)
                                      .with_alpha(alpha)
                                      .with_num_threads(1)
                                      .build();

        auto index_search_params = diskann::IndexSearchParams(index_build_params.search_list_size, index_build_params.num_threads);

        auto config = diskann::IndexConfigBuilder()
                          .with_metric(metric)
                          .with_dimension(data_dim)
                          .with_max_points(data_num)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .with_data_type(data_type)
                          .with_label_type(label_type)
                          .with_index_write_params(index_build_params)
                          .with_index_search_params(index_search_params)
                          .is_dynamic_index(true)
                          .is_enable_tags(true)
                          .is_use_opq(false)
                          .is_pq_dist_build(false)
                          .with_num_pq_chunks(0)
                          .is_concurrent_consolidate(false)
                          .build();

        auto index_factory = diskann::IndexFactory(config);
        auto index = index_factory.create_instance();
        index->set_start_points_at_random(static_cast<float>(0));
     
        diskann::Timer timer;
        //index->build(data, data_num, tags);
        // auto base_size = data_num;
        auto base_size = data_num;
        auto base_size_half = base_size / 2; // HALF
        auto base_size_fourth = base_size / 4;
        for (size_t i = 0; i < base_size; i++)
            index->insert_point(&data[i * aligned_dim], tags[i]);
        
        //for (size_t i = 0; i < base_size_fourth; i++)
        //{
        //    auto first_pos = 0 + i;
        //    index->insert_point(&data[first_pos * aligned_dim], tags[first_pos]);
        //    auto second_pos = base_size_half + i;
        //    index->insert_point(&data[second_pos * aligned_dim], tags[second_pos]);

        //    if ((i % (base_size_fourth / 10)) == 0) 
        //        std::cout << "added " << std::to_string(i * 2) << " after " << (timer.elapsed() / 1000000.0) << " seconds." << std::endl;
        //}

        //for (size_t i = 0; i < base_size_fourth; i++)
        //{
        //    auto first_pos = base_size_fourth + i;
        //    index->insert_point(&data[first_pos * aligned_dim], tags[first_pos]);
        //    auto second_pos = base_size_half + base_size_fourth + i;
        //    index->insert_point(&data[second_pos * aligned_dim], tags[second_pos]);

        //    index->lazy_delete(tags[base_size_half + (i * 2) + 0]);
        //    index->lazy_delete(tags[base_size_half + (i * 2) + 1]);

        //    if ((i % (base_size_fourth/10)) == 0) // 10%
        //    {
        //        index->consolidate_deletes(index_build_params);
        //        std::cout << "added " << std::to_string(base_size_half + i * 2) << ", deleted " << std::to_string(i)
        //                  << " after "
        //                  << (timer.elapsed() / 1000000.0) << " seconds." << std::endl;
        //    }

        //    //auto report = index->consolidate_deletes(index_build_params);
        //    //std::cout << "#active points: " << report._active_points << std::endl
        //    //          << "max points: " << report._max_points << std::endl
        //    //          << "empty slots: " << report._empty_slots << std::endl
        //    //          << "deletes processed: " << report._slots_released << std::endl
        //    //          << "latest delete size: " << report._delete_set_size << std::endl
        //    //          << "rate: (" << 0 / report._time << " points/second overall, "
        //    //          << 0 / report._time / 1 << " per thread)"
        //    //          << std::endl;
        //}
        //index->consolidate_deletes(index_build_params);
        //std::cout << std::to_string(base_size_half) << " elements in the index after " << (timer.elapsed() / 1000000.0) << " seconds." << std::endl;

        for (size_t i = 0; i < base_size_half; i++)
        {
            index->lazy_delete(tags[base_size_half + i]);
            if ((i % (base_size_half / 10)) == 0) // 10% 
            {
                index->consolidate_deletes(index_build_params);
                std::cout << std::to_string(i) << " deleted after " << (timer.elapsed() / 1000000.0) << " seconds." << std::endl;
            }
        }
        index->consolidate_deletes(index_build_params);
        std::cout << std::to_string(base_size_half) << " entries after " << (timer.elapsed() / 1000000.0) << " seconds." << std::endl;

        index->save(index_path_prefix.c_str(), true);
        //index->save(index_path_prefix.c_str());
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
