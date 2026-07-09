#include "read_pipeline/base/stitch.h"

#include "read_pipeline/base/messages/ReadCommon.h"
#include "utils/math_utils.h"

#include <ATen/ops/zeros.h>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define TEST_GROUP "[utils]"

// clang-format off
constexpr size_t RAW_SIGNAL_SIZE = 50;
const std::vector<std::string> SEQS(7, "ACGT");
const std::vector<std::string> QSTR(7, "!&.-");
const std::vector<std::vector<uint8_t>> MOVES{
        {1, 0, 0, 1, 0, 0, 1, 0, 1, 0}, 
        {1, 0, 0, 1, 0, 0, 0, 1, 0, 1},
        {1, 0, 0, 1, 0, 1, 1, 0, 0, 0},
        {1, 0, 0, 1, 0, 0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0, 0, 1, 0, 1, 0},
        {1, 0, 0, 0, 0, 0, 1, 0, 1, 1},
        {1, 0, 0, 1, 0, 0, 1, 0, 1, 0}};
/*
A        C        G     T
1, 0, 0, 1, 0, 0, 1, 0, 1, 0
                     A        C           G     T
                     1, 0, 0, 1, 0, 0, 0, 1, 0, 1
                                          A        C     G  T
                                          1, 0, 0, 1, 0, 1, 1, 0, 0, 0
                                                               A        C        G     T
                                                               1, 0, 0, 1, 0, 0, 1, 0, 1, 0
                                                                                       A     C        G     T
                                                                                    0, 1, 0, 1, 0, 0, 1, 0, 1, 0
                                                                                                         A                 C     G  T
                                                                                                         1, 0, 0, 0, 0, 0, 1, 0, 1, 1
                                                                                                                        A        C        G     T
                                                                                                                        1, 0, 0, 1, 0, 0, 1, 0, 1, 0
=
A        C        G     T     C           G        C     G  T           C        G     T     C        G     T              C     C        G     T
1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0
*/
// clang-format on

namespace {

dorado::ReadCommon make_read_common(const size_t raw_data_samples, const int model_stride) {
    dorado::ReadCommon read_common;
    read_common.raw_data = at::zeros({static_cast<int64_t>(raw_data_samples)});
    read_common.attributes.model_stride = model_stride;
    return read_common;
}

std::vector<const dorado::utils::Chunk *> make_chunk_ptrs(
        const std::vector<dorado::utils::Chunk> &called_chunks) {
    std::vector<const dorado::utils::Chunk *> chunks;
    chunks.reserve(called_chunks.size());
    for (const auto &chunk : called_chunks) {
        chunks.emplace_back(&chunk);
    }
    return chunks;
}

}  // namespace

CATCH_TEST_CASE("Test stitch_chunks", TEST_GROUP) {
    constexpr size_t CHUNK_SIZE = 10;
    constexpr size_t OVERLAP = 3;

    std::vector<std::unique_ptr<dorado::utils::Chunk>> called_chunks;

    size_t offset = 0;
    size_t signal_chunk_step = CHUNK_SIZE - OVERLAP;
    {
        auto chunk = std::make_unique<dorado::utils::Chunk>(offset, CHUNK_SIZE);
        const size_t chunk_idx = called_chunks.size();
        chunk->qstring = QSTR[chunk_idx];
        chunk->seq = SEQS[chunk_idx];
        chunk->moves = MOVES[chunk_idx];
        called_chunks.push_back(std::move(chunk));
    }
    while (offset + CHUNK_SIZE < RAW_SIGNAL_SIZE) {
        offset = std::min(offset + signal_chunk_step, RAW_SIGNAL_SIZE - CHUNK_SIZE);
        auto chunk = std::make_unique<dorado::utils::Chunk>(offset, CHUNK_SIZE);
        const size_t chunk_idx = called_chunks.size();
        chunk->qstring = QSTR[chunk_idx];
        chunk->seq = SEQS[chunk_idx];
        chunk->moves = MOVES[chunk_idx];
        called_chunks.push_back(std::move(chunk));
    }

    dorado::ReadCommon read_common;
    read_common.raw_data = at::zeros({static_cast<int64_t>(RAW_SIGNAL_SIZE)});
    read_common.attributes.model_stride = static_cast<int>(dorado::utils::div_round_closest(
            called_chunks[0]->raw_chunk_size, called_chunks[0]->moves.size()));

    std::vector<const dorado::utils::Chunk *> chunks;
    chunks.reserve(called_chunks.size());
    for (const auto &chunk : called_chunks) {
        chunks.emplace_back(chunk.get());
    }
    CATCH_REQUIRE_NOTHROW(dorado::utils::stitch_chunks(read_common, chunks, false));

    const std::string expected_sequence = "ACGTCGCGTCGTCGTCCGT";
    const std::string expected_qstring = "!&.-&.&.-&.-&.-&&.-";
    const std::vector<uint8_t> expected_moves = {1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
                                                 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                                                 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0};

    CATCH_REQUIRE(read_common.seq == expected_sequence);
    CATCH_REQUIRE(read_common.qstring == expected_qstring);
    CATCH_REQUIRE(read_common.moves == expected_moves);
}

CATCH_TEST_CASE("Test stitch_chunks trims non-full final chunks", TEST_GROUP) {
    constexpr int STRIDE = 2;

    CATCH_SECTION("single non-full chunk, non-VCS, with a base in the partial stride") {
        std::vector<dorado::utils::Chunk> called_chunks{{0, 10}};
        called_chunks[0].seq = "ACGT";
        called_chunks[0].qstring = "abcd";
        called_chunks[0].moves = {1, 1, 0, 1, 1};

        auto read_common = make_read_common(7, STRIDE);
        auto chunks = make_chunk_ptrs(called_chunks);
        CATCH_REQUIRE_NOTHROW(dorado::utils::stitch_chunks(read_common, chunks, false));

        CATCH_CHECK(read_common.seq == "AC");
        CATCH_CHECK(read_common.qstring == "ab");
        CATCH_CHECK(read_common.moves == std::vector<uint8_t>{1, 1, 0});
    }

    CATCH_SECTION("multiple VCS chunks trim the padded tail of the final chunk") {
        std::vector<dorado::utils::Chunk> called_chunks{{0, 8, 4}, {6, 4, 8}};
        called_chunks[0].seq = "AAAA";
        called_chunks[0].qstring = "!!!!";
        called_chunks[0].moves = {1, 1, 1, 1};
        called_chunks[1].seq = "WXYZ";
        called_chunks[1].qstring = "wxyz";
        called_chunks[1].moves = {1, 1, 1, 1};

        auto read_common = make_read_common(10, STRIDE);
        auto chunks = make_chunk_ptrs(called_chunks);
        CATCH_REQUIRE_NOTHROW(dorado::utils::stitch_chunks(read_common, chunks, true));

        CATCH_CHECK(read_common.seq == "AAAAX");
        CATCH_CHECK(read_common.qstring == "!!!!x");
        CATCH_CHECK(read_common.moves == std::vector<uint8_t>{1, 1, 1, 1, 1});
    }

    CATCH_SECTION("single non-full chunk, non-VCS, with no base in the partial stride") {
        std::vector<dorado::utils::Chunk> called_chunks{{0, 10}};
        called_chunks[0].seq = "ACG";
        called_chunks[0].qstring = "abc";
        called_chunks[0].moves = {1, 1, 0, 0, 1};

        auto read_common = make_read_common(7, STRIDE);
        auto chunks = make_chunk_ptrs(called_chunks);
        CATCH_REQUIRE_NOTHROW(dorado::utils::stitch_chunks(read_common, chunks, false));

        CATCH_CHECK(read_common.seq == "AC");
        CATCH_CHECK(read_common.qstring == "ab");
        CATCH_CHECK(read_common.moves == std::vector<uint8_t>{1, 1, 0});
    }
}
