#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <input.bam>"
    exit 1
fi

if ! command -v samtools >/dev/null 2>&1; then
    echo "Error: samtools is required but was not found in PATH."
    exit 1
fi

input_bam=$1

if [[ ! -f "${input_bam}" ]]; then
    echo "Error: input BAM does not exist: ${input_bam}"
    exit 1
fi

samtools view "${input_bam}" | awk '
{
    qname = $1
    parent_id = ""
    sp = ""
    ts = ""
    ns = ""

    for (i = 12; i <= NF; ++i) {
        if ($i ~ /^pi:Z:/) {
            parent_id = substr($i, 6)
        } else if ($i ~ /^sp:i:/) {
            sp = substr($i, 6) + 0
        } else if ($i ~ /^ns:i:/) {
            ns = substr($i, 6) + 0
        }
    }

    if (sp == "" && parent_id == "") {
        next
    }

    if (parent_id == "") {
        printf("Validation failed for read %s: missing pi tag for split read\n", qname) > "/dev/stderr"
        exit 1
    }

    if (sp == "") {
        printf("Validation failed for read %s: missing sp tag\n", qname) > "/dev/stderr"
        exit 1
    }

    if (ns == "") {
        printf("Validation failed for read %s: missing ns tag\n", qname) > "/dev/stderr"
        exit 1
    }

    print parent_id "\t" sp "\t" ns "\t" qname
}
' | sort -t $'\t' -k1,1 -k2,2n -k4,4 | awk '
BEGIN {
    split_read_count = 0
    validation_failed = 0
}

{
    parent_id = $1
    sp = $2 + 0
    ns = $3 + 0
    qname = $4
    range_start = sp
    range_end = sp + ns
    split_read_count++

    if (seen_parent[parent_id] && prev_end[parent_id] > range_start) {
        validation_failed = 1
        printf("Validation failed for parent %s: read %s range [%d, %d) overlaps read %s range [%d, %d)\n",
               parent_id, prev_qname[parent_id], prev_start[parent_id], prev_end[parent_id],
               qname, range_start, range_end) > "/dev/stderr"
        exit 1
    }

    seen_parent[parent_id] = 1
    prev_qname[parent_id] = qname
    prev_start[parent_id] = range_start
    prev_end[parent_id] = range_end
}

END {
    if (validation_failed) {
        exit 1
    }

    if (split_read_count == 0) {
        print "No split reads with pi/sp tags found; validation passed."
        exit 0
    }

    printf("Validation passed for %d split reads.\n", split_read_count)
}
'
