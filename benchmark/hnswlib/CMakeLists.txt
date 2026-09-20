cmake_minimum_required(VERSION 3.19)

project(benchmark)
include_directories(${PROJECT_SOURCE_DIR}/include)
set(DATA_PATH "" CACHE PATH "Path to data sets")

if(DATA_PATH STREQUAL "")
    message(WARNING "DATA_PATH is empty")
endif()

add_executable(hnswlib_build_and_test ${PROJECT_SOURCE_DIR}/src/hnswlib_build_and_test.cpp)
target_link_libraries(hnswlib_build_and_test PUBLIC HNSW_LIB compile-options fmt::fmt-header-only)
target_compile_definitions(hnswlib_build_and_test PUBLIC "DATA_PATH=\"${DATA_PATH}\"")
