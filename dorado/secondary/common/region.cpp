#include "secondary/common/region.h"

#include "utils/string_utils.h"

#include <IntervalTree.h>

#include <filesystem>
#include <fstream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace dorado::secondary {

std::ostream& operator<<(std::ostream& os, const Region& region) {
    os << region.name << ":" << (region.start + 1) << "-" << region.end;
    return os;
}

bool operator<(const Region& l, const Region& r) {
    return std::tie(l.name, l.start, l.end) < std::tie(r.name, r.start, r.end);
}

std::string to_string(const Region& region) {
    std::ostringstream oss;
    oss << region;
    return std::move(oss).str();
}

std::ostream& operator<<(std::ostream& os, const RegionInt& region) {
    os << region.seq_id << ':' << (region.start + 1) << '-' << region.end;
    return os;
}

std::string to_string(const RegionInt& region) {
    std::ostringstream oss;
    oss << region;
    return std::move(oss).str();
}

bool is_valid(const RegionInt& region) {
    if ((region.seq_id < 0) || (region.start < -1) || (region.end < -1)) {
        return false;
    }
    return (region.start < 0) || (region.end < 0) || (region.start < region.end);
}

RegionInt normalize_region(const RegionInt& region, const int64_t seq_len) {
    if (!is_valid(region)) {
        throw std::runtime_error{"Cannot normalize a region that is not valid. Given: " +
                                 to_string(region)};
    }
    return RegionInt{
            .seq_id = region.seq_id,
            .start = std::clamp<int64_t>(region.start, 0, seq_len),
            .end = (region.end < 0) ? seq_len : std::clamp<int64_t>(region.end, 0, seq_len),
    };
}

Region parse_region_string(const std::string& region) {
    const size_t colon_pos = region.find(':');
    if (colon_pos == std::string::npos) {
        return {region, -1, -1};
    }

    std::string name = region.substr(0, colon_pos);

    if ((colon_pos + 1) == std::size(region)) {
        return {std::move(name), -1, -1};
    }

    size_t dash_pos = region.find('-', colon_pos + 1);
    dash_pos = (dash_pos == std::string::npos) ? std::size(region) : dash_pos;
    const int64_t start =
            ((dash_pos - colon_pos - 1) == 0)
                    ? -1
                    : std::stoll(region.substr(colon_pos + 1, dash_pos - colon_pos - 1)) - 1;
    const int64_t end =
            ((dash_pos + 1) < std::size(region)) ? std::stoll(region.substr(dash_pos + 1)) : -1;

    return Region{std::move(name), start, end};
}

std::vector<Region> parse_regions(const std::string& regions_arg) {
    if (std::empty(regions_arg)) {
        return {};
    }

    std::vector<Region> ret;

    // Check if the string points to a file on disk.
    if (std::filesystem::exists(regions_arg)) {
        // Parse the BED-like format.
        std::ifstream ifs(regions_arg);
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::string chr;
            int64_t start = 0;
            int64_t end = 0;
            iss >> chr >> start >> end;
            ret.emplace_back(Region{chr, start, end});
        }

    } else {
        // Parse a comma delimited string of regions.
        const auto str_regions = utils::split(regions_arg, ',');
        for (const std::string& str_region : str_regions) {
            Region region = parse_region_string(str_region);
            ret.emplace_back(region);
        }
    }

    return ret;
}

void validate_regions(const std::vector<Region>& regions,
                      const std::vector<std::pair<std::string, int64_t>>& seq_lens) {
    // Create intervals for each input sequence.
    std::unordered_map<std::string, std::vector<interval_tree::Interval<int64_t, int64_t>>>
            intervals;
    for (const auto& region : regions) {
        // NOTE: interval_tree has an inclusive end coordinate.
        intervals[region.name].emplace_back(
                interval_tree::Interval<int64_t, int64_t>(region.start, region.end - 1, 0));
    }

    // Compute the interval tree.
    std::unordered_map<std::string, interval_tree::IntervalTree<int64_t, int64_t>> trees;
    for (auto& [key, values] : intervals) {
        trees[key] = interval_tree::IntervalTree<int64_t, int64_t>(std::move(values));
    }

    // Validate that none of the regions is overlapping any other region.
    for (const auto& region : regions) {
        std::vector<interval_tree::Interval<int64_t, int64_t>> results =
                trees[region.name].findOverlapping(region.start, region.end - 1);
        if (std::size(results) > 1) {
            throw std::runtime_error("Region validation failed: region '" + to_string(region) +
                                     "' overlaps other regions. Regions have to be unique.");
        }
    }

    // Validate that all of the regions are within the range of the input sequences.
    std::unordered_map<std::string, int64_t> len_dict;
    for (const auto& [key, val] : seq_lens) {
        len_dict[key] = val;
    }
    for (const auto& region : regions) {
        const auto it = len_dict.find(region.name);
        if (it == std::end(len_dict)) {
            throw std::runtime_error{"Region validation failed: sequence name for region '" +
                                     to_string(region) +
                                     "' does not exist in the input sequence file."};
        }
        const int64_t seq_len = it->second;
        // Allow negative coordinates as a proxy for full sequence length.
        if ((region.start >= seq_len) || (region.end > seq_len) ||
            ((region.start >= 0) && (region.end >= 0) && (region.start >= region.end))) {
            throw std::runtime_error{
                    "Region validation failed: coordinates for region '" + to_string(region) +
                    "' are not valid. Sequence length: " + std::to_string(seq_len)};
        }
    }
}

}  // namespace dorado::secondary
