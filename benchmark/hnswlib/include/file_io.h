#pragma once

/**
 * @file file_io.h
 * @brief File I/O utilities for vector files (fvecs, ivecs format) and filesystem operations.
 */

#include <fmt/core.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnswlib::benchmark {

// ============================================================================
// Vector Repository
// ============================================================================

/**
 * A simple repository for vectors, replacing deglib::FeatureRepository.
 * Stores vectors in a contiguous memory block.
 */
class VectorRepository {
    std::vector<float> data_;
    size_t dim_;
    size_t count_;

public:
    VectorRepository() : dim_(0), count_(0) {}

    // Make move constructible
    VectorRepository(VectorRepository&&) = default;
    VectorRepository& operator=(VectorRepository&&) = default;

    // Disable copy
    VectorRepository(const VectorRepository&) = delete;
    VectorRepository& operator=(const VectorRepository&) = delete;

    void resize(size_t count, size_t dim) {
        count_ = count;
        dim_ = dim;
        data_.resize(count * dim);
    }

    const float* getFeature(size_t idx) const { return &data_[idx * dim_]; }

    // Access raw data
    size_t size() const { return count_; }
    size_t dims() const { return dim_; }

    const float* data() const { return data_.data(); }
    float* data() { return data_.data(); }
};

// ============================================================================
// Vector File I/O (fvecs, ivecs)
// ============================================================================

/**
 * Read ivecs into a uint32_t buffer.
 *
 * @param filename Input file path
 * @param d_out Dimension output
 * @param n_out Number of vectors output
 */
inline std::unique_ptr<uint32_t[]> ivecs_read(const char* filename, size_t& d_out, size_t& n_out) {
    d_out = 0;
    n_out = 0;

    std::error_code ec{};
    const auto file_size = std::filesystem::file_size(filename, ec);
    if (ec != std::error_code{}) {
        fmt::print(stderr, "Error accessing ivecs file '{}': {}\n", filename, ec.message());
        return nullptr;
    }

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        fmt::print(stderr, "Could not open ivecs file '{}'.\n", filename);
        return nullptr;
    }

    uint32_t dims = 0;
    in.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    if (!in) {
        fmt::print(stderr, "Could not read ivecs header from '{}'.\n", filename);
        return nullptr;
    }

    if (dims == 0 || dims > 1'000'000) {
        fmt::print(stderr, "Unreasonable ivecs dimension {} in '{}'.\n", dims, filename);
        return nullptr;
    }

    const size_t row_bytes = (static_cast<size_t>(dims) + 1) * sizeof(uint32_t);
    if (row_bytes == 0 || (file_size % row_bytes) != 0) {
        fmt::print(stderr,
                   "Weird ivecs file size for '{}': {} bytes not divisible by row size {} (dims={}).\n",
                   filename,
                   static_cast<uintmax_t>(file_size),
                   row_bytes,
                   dims);
        return nullptr;
    }

    const size_t n = static_cast<size_t>(file_size / row_bytes);
    d_out = dims;
    n_out = n;

    auto file_bytes = std::make_unique<std::byte[]>(static_cast<size_t>(file_size));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(file_bytes.get()), static_cast<std::streamsize>(file_size));
    if (!in) {
        fmt::print(stderr, "Could not read whole ivecs file '{}'.\n", filename);
        return nullptr;
    }

    auto data = std::make_unique<uint32_t[]>(n_out * d_out);
    for (size_t i = 0; i < n_out; ++i) {
        const std::byte* row = file_bytes.get() + i * row_bytes;
        uint32_t dim_check = 0;
        std::memcpy(&dim_check, row, sizeof(uint32_t));
        if (dim_check != dims) {
            fmt::print(stderr, "ivecs dimension mismatch in '{}': row {} has dim {} but expected {}.\n", filename, i, dim_check, dims);
            d_out = 0;
            n_out = 0;
            return nullptr;
        }

        std::memcpy(data.get() + i * d_out, row + sizeof(uint32_t), d_out * sizeof(uint32_t));
    }

    return data;
}

/**
 * Load vectors from fvecs file into a VectorRepository.
 */
inline VectorRepository load_static_repository(const char* filename) {
    VectorRepository repo;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(std::string("Could not open file: ") + filename);
    }

    int dim;
    in.read((char*)&dim, 4);
    if (!in) return repo;

    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    size_t vec_size = 4 + dim * 4;
    size_t n = file_size / vec_size;

    repo.resize(n, dim);

    float* data_ptr = repo.data();
    for (size_t i = 0; i < n; ++i) {
        int d;
        in.read((char*)&d, 4);
        if (d != dim) throw std::runtime_error("Dimension mismatch in file during load");
        in.read((char*)(data_ptr + i * dim), dim * 4);
    }

    return repo;
}

// ============================================================================
// Filesystem Utilities
// ============================================================================

/**
 * Check if a file or directory exists.
 */
inline bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

inline bool file_exists(const std::filesystem::path& path) {
    return std::filesystem::exists(path);
}

/**
 * Ensure a directory exists, creating it if necessary.
 * @return true if directory exists or was created successfully
 */
inline bool ensure_directory(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) return true;
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        fmt::print(stderr, "Error creating directory '{}': {}\n", path.string(), ec.message());
        return false;
    }
    return true;
}

/**
 * Delete a file if it exists.
 * @return true if file was deleted or didn't exist
 */
inline bool delete_file(const std::string& path) {
    if (!std::filesystem::exists(path)) return true;
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec) {
        fmt::print(stderr, "Error deleting file '{}': {}\n", path, ec.message());
        return false;
    }
    fmt::print("Deleted: {}\n", path);
    return true;
}

inline bool delete_file(const std::filesystem::path& path) {
    return delete_file(path.string());
}

/**
 * Rename/move a file.
 * @return true if successful
 */
inline bool rename_file(const std::string& from, const std::string& to) {
    if (!std::filesystem::exists(from)) {
        return false;  // Source doesn't exist, not an error - file may be optional
    }
    std::error_code ec;
    std::filesystem::rename(from, to, ec);
    if (ec) {
        fmt::print(stderr, "Error renaming '{}' to '{}': {}\n", from, to, ec.message());
        return false;
    }
    fmt::print("Renamed: {} -> {}\n", from, to);
    return true;
}

inline bool rename_file(const std::filesystem::path& from, const std::filesystem::path& to) {
    return rename_file(from.string(), to.string());
}

/**
 * Move a file from source to destination, with optional renaming.
 * Creates destination directory if needed.
 * @param src Source file path
 * @param dest Destination file path
 * @return true if file was moved successfully, false if source doesn't exist or error occurred
 */
inline bool move_file(const std::filesystem::path& src, const std::filesystem::path& dest) {
    if (!std::filesystem::exists(src)) {
        return false;  // Source doesn't exist
    }

    // Ensure destination directory exists
    ensure_directory(dest.parent_path());

    std::error_code ec;
    std::filesystem::rename(src, dest, ec);
    if (ec) {
        fmt::print(stderr, "Error moving '{}' to '{}': {}\n", src.string(), dest.string(), ec.message());
        return false;
    }
    fmt::print("Moved: {} -> {}\n", src.string(), dest.string());
    return true;
}

/**
 * Remove a directory and all its contents.
 * @return true if directory was removed or didn't exist
 */
inline bool remove_directory(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) return true;
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec) {
        fmt::print(stderr, "Error removing directory '{}': {}\n", path.string(), ec.message());
        return false;
    }
    return true;
}

// ============================================================================
// Download and Extract Utilities (Removed)
// ============================================================================

}  // namespace hnswlib::benchmark
