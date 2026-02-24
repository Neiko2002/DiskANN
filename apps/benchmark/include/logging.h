#pragma once

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <streambuf>
#include <string>
#include <utility>

namespace diskann::benchmark
{

inline std::ofstream log_file_stream;
inline bool log_to_console = true;
inline std::streambuf *cout_buffer_backup = nullptr;
inline std::streambuf *cerr_buffer_backup = nullptr;

class TeeBuf : public std::streambuf
{
  public:
    TeeBuf(std::streambuf *sb1, std::streambuf *sb2) : sb1_(sb1), sb2_(sb2)
    {
    }

  protected:
    int overflow(int c) override
    {
        if (c == EOF)
        {
            return !EOF;
        }
        const int r1 = sb1_ ? sb1_->sputc(c) : c;
        const int r2 = sb2_ ? sb2_->sputc(c) : c;
        return (r1 == EOF || r2 == EOF) ? EOF : c;
    }

    int sync() override
    {
        int r1 = sb1_ ? sb1_->pubsync() : 0;
        int r2 = sb2_ ? sb2_->pubsync() : 0;
        return (r1 == 0 && r2 == 0) ? 0 : -1;
    }

  private:
    std::streambuf *sb1_;
    std::streambuf *sb2_;
};

inline std::unique_ptr<TeeBuf> cout_tee_buf;
inline std::unique_ptr<TeeBuf> cerr_tee_buf;

inline void set_log_file(const std::string &path, bool append = false)
{
    if (log_file_stream.is_open())
    {
        log_file_stream.close();
    }
    auto mode = std::ios::out;
    if (append)
    {
        mode |= std::ios::app;
    }
    log_file_stream.open(path, mode);
    if (!log_file_stream.is_open())
    {
        std::cerr << "Warning: Could not open log file '" << path << "'\n";
    }
}

inline void attach_cerr_to_log()
{
    if (!log_file_stream.is_open())
    {
        return;
    }
    if (cerr_buffer_backup == nullptr)
    {
        cerr_buffer_backup = std::cerr.rdbuf();
        cerr_tee_buf = std::make_unique<TeeBuf>(cerr_buffer_backup, log_file_stream.rdbuf());
        std::cerr.rdbuf(cerr_tee_buf.get());
    }
}

inline void detach_cerr_from_log()
{
    if (cerr_buffer_backup != nullptr)
    {
        std::cerr.rdbuf(cerr_buffer_backup);
        cerr_buffer_backup = nullptr;
        cerr_tee_buf.reset();
    }
}

inline void attach_cout_to_log()
{
    if (!log_file_stream.is_open())
    {
        return;
    }
    if (cout_buffer_backup == nullptr)
    {
        cout_buffer_backup = std::cout.rdbuf();
        cout_tee_buf = std::make_unique<TeeBuf>(cout_buffer_backup, log_file_stream.rdbuf());
        std::cout.rdbuf(cout_tee_buf.get());
    }
}

inline void detach_cout_from_log()
{
    if (cout_buffer_backup != nullptr)
    {
        std::cout.rdbuf(cout_buffer_backup);
        cout_buffer_backup = nullptr;
        cout_tee_buf.reset();
    }
}

inline void reset_log_to_console()
{
    detach_cerr_from_log();
    detach_cout_from_log();
    if (log_file_stream.is_open())
    {
        log_file_stream.close();
    }
}

inline void set_console_logging(bool enabled)
{
    log_to_console = enabled;
}

inline std::string string_format(const char *fmt)
{
    return std::string(fmt);
}

template <typename... Args> inline std::string string_format(const char *fmt, Args &&...args)
{
    const int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    if (size <= 0)
    {
        return {};
    }
    std::string buf;
    buf.resize(static_cast<size_t>(size));
    std::snprintf(&buf[0], static_cast<size_t>(size) + 1, fmt, std::forward<Args>(args)...);
    return buf;
}

inline void log(const char *msg)
{
    if (cout_buffer_backup != nullptr)
    {
        std::cout << msg;
        std::cout.flush();
    }
    else
    {
        if (log_to_console)
        {
            std::cout << msg;
            std::cout.flush();
        }
        if (log_file_stream.is_open())
        {
            log_file_stream << msg;
            log_file_stream.flush();
        }
    }
}

template <typename... Args> inline void log(const char *fmt, Args &&...args)
{
    const std::string msg = string_format(fmt, std::forward<Args>(args)...);

    if (cout_buffer_backup != nullptr)
    {
        std::cout << msg;
        std::cout.flush();
    }
    else
    {
        if (log_to_console)
        {
            std::cout << msg;
            std::cout.flush();
        }
        if (log_file_stream.is_open())
        {
            log_file_stream << msg;
            log_file_stream.flush();
        }
    }
}

} // namespace diskann::benchmark
