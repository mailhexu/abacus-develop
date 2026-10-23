#ifndef ABACUS_TB2J_SHA256_H
#define ABACUS_TB2J_SHA256_H

#include <cstddef>
#include <string>

namespace ModuleIO
{
namespace Tb2j
{

/// @brief SHA-256 digest of len bytes, as 64 lowercase hex characters.
/// @param data input bytes (may be null when len == 0)
/// @param len number of input bytes
std::string sha256_hex(const void* data, std::size_t len);

/// @brief SHA-256 digest of a file's raw bytes, streamed in fixed blocks.
/// @param path file to hash
/// @return 64 lowercase hex characters; empty string if the file is unreadable
std::string sha256_file_hex(const std::string& path);

} // namespace Tb2j
} // namespace ModuleIO

#endif // ABACUS_TB2J_SHA256_H
