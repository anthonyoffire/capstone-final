#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
MODE_ARGS=()
APPEND_ARGS=()

while (($#)); do
    case "$1" in
        --tags-only)
            MODE_ARGS+=(--tags-only)
            ;;
        --append-tags)
            APPEND_ARGS+=(--append-tags)
            ;;
        *)
            echo "Usage: $0 [--tags-only] [--append-tags]" >&2
            exit 2
            ;;
    esac
    shift
done

run_job() {
    local job_number="$1"
    shift
    local start_age="$1"
    local end_age="$2"
    local images_per_age="$3"
    local image_counter="$4"

    "$PYTHON_BIN" ./dataset/generation/diffusion_gen.py \
        "${MODE_ARGS[@]}" "${APPEND_ARGS[@]}" \
        -s "$start_age" -e "$end_age" -n "$images_per_age" -c "$image_counter" \
        --tags-file "tags_job${job_number}.csv"
}

run_job 1 1 99 250 0
run_job 2 1 99 250 519750
run_job 3 1 99 250 1039500
run_job 4 1 99 250 1559250
run_job 5 1 24 1000 2079000
run_job 6 50 99 500 2583000
run_job 7 50 99 500 3108000
run_job 8 1 17 1000 3633000
run_job 9 1 15 1500 3990000
run_job 10 1 13 1500 4462500
run_job 11 59 99 1000 4872000
run_job 12 63 99 1000 5733000
run_job 13 1 7 500 6510000
run_job 14 55 62 500 6583500
run_job 15 55 99 500 6667500
run_job 16 66 99 500 7140000
