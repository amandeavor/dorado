#include "secondary/common/batching.h"

#include <stdexcept>

namespace dorado::secondary {

std::vector<Interval> compute_partitions(const int32_t num_items, const int32_t num_partitions) {
    std::vector<Interval> chunks;
    const int32_t chunk_size = num_items / num_partitions;
    std::vector<int32_t> chunk_sizes(num_partitions, chunk_size);
    for (int32_t i = 0; i < (num_items % num_partitions); ++i) {
        ++chunk_sizes[i];
    }
    int32_t sum = 0;
    for (const int32_t v : chunk_sizes) {
        if (v == 0) {
            continue;
        }
        chunks.emplace_back(Interval{sum, sum + v});
        sum += v;
    }
    if (sum != num_items) {
        throw std::runtime_error{
                "Wrong sum of items divided into chunks! num_items = " + std::to_string(num_items) +
                ", num_partitions = " + std::to_string(num_partitions) +
                ", sum = " + std::to_string(sum)};
    }
    return chunks;
}

std::pair<std::vector<std::vector<Region>>, std::vector<Interval>> prepare_region_batches(
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& ref_lookup,
        const std::vector<std::pair<std::string, int64_t>>& bam_ref_seqs,
        const std::vector<Region>& user_regions,
        const int64_t draft_batch_size) {
    // Outer vector: ID of the draft, inner vector: regions.
    std::vector<std::vector<Region>> ret(std::size(ref_lookup));

    if (std::empty(user_regions)) {
        // Add full draft sequences referenced in the input BAM.
        for (const auto& [ref_name, ref_len_from_bam] : bam_ref_seqs) {
            const auto it = ref_lookup.find(ref_name);
            if (it == std::cend(ref_lookup)) {
                throw std::runtime_error{
                        "BAM header references a sequence which is not present in the input "
                        "reference FASTA file. Sequence name: '" +
                        ref_name + "'"};
            }
            const auto [ref_id, ref_len] = it->second;
            if (ref_len != ref_len_from_bam) {
                throw std::runtime_error{
                        "Length of the reference sequence differs between the input reference "
                        "FASTA and the BAM header. Sequence name: '" +
                        ref_name + "', length from FASTA: " + std::to_string(ref_len) +
                        ", length from BAM: " + std::to_string(ref_len_from_bam)};
            }
            ret[ref_id].emplace_back(Region{ref_name, 0, ref_len});
        }

    } else {
        // Bin the user regions for individual contigs.
        for (const auto& region : user_regions) {
            const auto it = ref_lookup.find(region.name);
            if (it == std::cend(ref_lookup)) {
                throw std::runtime_error(
                        "Sequence name from a custom specified region not found in the input "
                        "sequence file! region: " +
                        to_string(region));
            }
            const auto [ref_id, ref_len] = it->second;
            ret[ref_id].emplace_back(Region{region.name, region.start, region.end});
        }
    }

    // Divide draft sequences into groups of specified size, as sort of a barrier.
    std::vector<Interval> region_batches =
            create_batches(ret, draft_batch_size, [](const std::vector<Region>& regions) {
                int64_t sum = 0;
                for (const auto& region : regions) {
                    sum += region.end - region.start;
                }
                return sum;
            });

    return std::make_pair(std::move(ret), std::move(region_batches));
}

}  // namespace dorado::secondary
