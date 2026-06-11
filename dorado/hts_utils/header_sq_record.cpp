#include "hts_utils/header_sq_record.h"

#include "utils/sequence_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <numeric>
#include <vector>

namespace {
const std::vector<unsigned char> to_upper = []() {
    std::vector<unsigned char> lookup(256);
    std::iota(std::begin(lookup), std::end(lookup), 0);
    constexpr char offset = 'A' - 'a';
    for (char i = 'a'; i <= 'z'; ++i) {
        lookup[i] = i + offset;
    }
    return lookup;
}();
}  // namespace

namespace dorado::utils {

void add_sq_hdr(sam_hdr_t* hdr, const HeaderSQRecords& seqs) {
    for (const auto& s : seqs) {
        if (s.uri != nullptr) {
            sam_hdr_add_line(hdr, "SQ", "SN", s.sequence_name.c_str(), "LN",
                             std::to_string(s.length).c_str(), "M5", s.md5, "UR", s.uri->c_str(),
                             NULL);
        } else {
            sam_hdr_add_line(hdr, "SQ", "SN", s.sequence_name.c_str(), "LN",
                             std::to_string(s.length).c_str(), NULL);
        }
    }
}

MD5Generator::MD5Generator() : m_ctx(hts_md5_init()) {}
MD5Generator::~MD5Generator() { hts_md5_destroy(m_ctx); }

void MD5Generator::get_sequence_md5(MD5Hex& hex, std::string_view sequence) {
    hts_md5_reset(m_ctx);
    std::string transformed_sequence(sequence.size(), 0);
    for (size_t idx = 0; idx < sequence.size(); ++idx) {
        transformed_sequence[idx] = to_upper[sequence[idx]];
    }

    std::erase_if(transformed_sequence, [](char base) {
        unsigned char c = static_cast<unsigned char>(base);
        return c < 33 || c > 126;
    });

    hts_md5_update(m_ctx, transformed_sequence.data(),
                   static_cast<uint32_t>(transformed_sequence.size()));
    unsigned char digest[16];
    hts_md5_final(digest, m_ctx);
    hts_md5_hex(hex, digest);
}

}  // namespace dorado::utils