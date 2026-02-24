# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# Define a list of all executables (tools)
set(DISKANN_TOOLS
    dynamic_memory_index
    explore_memory_index
    build_memory_index
    build_stitched_index
    search_memory_index
    build_disk_index
    search_disk_index
    range_search_disk_index
    test_streaming_scenario
    test_insert_deletes_consolidate
    diskann_build_and_test
)

# Since we downgraded to vc142 packages, disable MSVC auto-linking for Boost
if (MSVC)
    add_compile_definitions(BOOST_ALL_NO_LIB)
endif()

add_executable(dynamic_memory_index dynamic_memory_index.cpp)
target_link_libraries(dynamic_memory_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(explore_memory_index explore_memory_index.cpp)
target_link_libraries(explore_memory_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(build_stitched_index build_stitched_index.cpp)
target_link_libraries(build_stitched_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(search_memory_index search_memory_index.cpp)
target_link_libraries(search_memory_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} ${DISKANN_ASYNC_LIB} Boost::program_options)

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(range_search_disk_index range_search_disk_index.cpp)
target_link_libraries(range_search_disk_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(test_streaming_scenario test_streaming_scenario.cpp)
target_link_libraries(test_streaming_scenario ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(test_insert_deletes_consolidate test_insert_deletes_consolidate.cpp)
target_link_libraries(test_insert_deletes_consolidate ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(diskann_build_and_test benchmark/src/diskann_build_and_test.cpp)
target_include_directories(diskann_build_and_test PRIVATE benchmark/include)
target_link_libraries(diskann_build_and_test ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)
target_compile_definitions(diskann_build_and_test PUBLIC
    "DATA_PATH=\"${DATA_PATH}\"")



if (NOT MSVC)
    install(TARGETS build_disk_index
        build_memory_index
        build_stitched_index
        dynamic_memory_index
        explore_memory_index
        range_search_disk_index
        search_disk_index
        search_memory_index
        test_insert_deletes_consolidate
        test_streaming_scenario        
        diskann_build_and_test
        RUNTIME
    )
endif()
