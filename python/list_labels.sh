#!/bin/bash

# Usage: ./list_labels.sh ./datasets_audio/raw_data standing_still walking_forward ...

dir=${1:-.}
shift
requested_labels=("$@")

# Check directory
if [[ ! -d "$dir" ]]; then
    echo "Error: Directory '$dir' does not exist." >&2
    exit 1
fi

# Get column indices from header
first_file=$(find "$dir" -name '*.csv' | head -n 1)
header=$(head -n 1 "$first_file")
IFS=',' read -r -a cols <<< "$header"

label_col_idx=-1
person_col_idx=-1
for i in "${!cols[@]}"; do
    colname=$(echo "${cols[i]}" | tr -d '\r\n')
    if [[ "$colname" == "label" ]]; then
        label_col_idx=$((i + 1))
    elif [[ "$colname" == "person" ]]; then
        person_col_idx=$((i + 1))
    fi
done

if [[ "$label_col_idx" == -1 || "$person_col_idx" == -1 ]]; then
    echo "Error: Could not find 'label' or 'person' column." >&2
    exit 1
fi

# Auto-detect labels if none specified
if [[ ${#requested_labels[@]} -eq 0 ]]; then
    mapfile -t requested_labels < <(awk -F, -v l="$label_col_idx" 'NR > 1 {print $l}' "$dir"/*.csv | sort -u)
fi

# Create temp file: one line per "person|label"
tmpfile=$(mktemp)
for file in "$dir"/*.csv; do
    awk -F, -v p="$person_col_idx" -v l="$label_col_idx" 'NR > 1 {print $p "|" $l}' "$file" >> "$tmpfile"
done

# Final aggregation using flat key person|label + grand total
awk -v labels="${requested_labels[*]}" '
BEGIN {
    split(labels, lbls, " ")
}
{
    split($0, parts, "|")
    person = parts[1]
    label = parts[2]
    key = person "|" label
    count[key]++
    people[person] = 1
    total[label]++
}
END {
    # Print header
    printf "%-10s", "Person"
    for (i in lbls) {
        printf "%15s", lbls[i]
    }
    print ""

    PROCINFO["sorted_in"] = "@ind_num_asc"
    for (p in people) {
        printf "%-10s", p
        for (i in lbls) {
            l = lbls[i]
            k = p "|" l
            printf "%15d", count[k] + 0
        }
        print ""
    }

    # Print total row
    printf "%-10s", "TOTAL"
    for (i in lbls) {
        l = lbls[i]
        printf "%15d", total[l] + 0
    }
    print ""
}
' "$tmpfile"

rm "$tmpfile"
