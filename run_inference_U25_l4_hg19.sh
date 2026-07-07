#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch inference/deconvolution runner for Syto
# Adapted to the new inference config template that uses:
#   input.type: parsed_reads
#   input.data_path: <per-sample reads CSV>
#   output_dir:      <top-level per-sample output dir>
#
# Iterates over every *_reads.csv in INPUT_DIR, generates a
# per-sample temp config from the template, and runs inference.
# ============================================================

CONFIG_TEMPLATE="App/config/inference/inference_20260623_150304.yaml"
TEMP_CONFIG_DIR="App/config/inference/tmp_configs"

INPUT_DIR="/mnt/data/cfsort/recovered_reads/U250.l4.hg19"
OUTPUT_BASE_DIR="/mnt/data/cfsort/deconvolution_results/U25.l4.hg19"

# Set to 1 to skip samples whose output already has deconvolution results.
SKIP_IF_DONE=1
DONE_MARKER_NAME="deconvolution_results.csv"

mkdir -p "$TEMP_CONFIG_DIR"

if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "ERROR: Config template not found: $CONFIG_TEMPLATE"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input dir does not exist: $INPUT_DIR"
    exit 1
fi

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR%/}"

echo "Template:    $CONFIG_TEMPLATE"
echo "Input dir:   $INPUT_DIR"
echo "Output base: $OUTPUT_BASE_DIR"
echo ""

# ---- Main loop -------------------------------------------------------------
n_total=0
n_done=0
n_failed=0
n_skipped=0

for input_path in "$INPUT_DIR"/*_reads.csv; do
    [ -f "$input_path" ] || continue
    n_total=$((n_total + 1))

    filename=$(basename "$input_path")
    sample_name="${filename%.csv}"
    sample_name="${sample_name%_reads}"

    output_dir="${OUTPUT_BASE_DIR}/${sample_name}"
    temp_config="${TEMP_CONFIG_DIR}/inference_${sample_name}.yaml"

    if [ "$SKIP_IF_DONE" -eq 1 ] && [ -f "${output_dir}/${DONE_MARKER_NAME}" ]; then
        echo "[SKIP] $sample_name (already done)"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    echo "============================================"
    echo "Processing: $sample_name"
    echo "  Input:      $input_path"
    echo "  Output dir: $output_dir"
    echo "============================================"

    # Replace input.data_path (indented) and top-level output_dir.
    sed \
        -e "s|^\(\s*data_path:\).*|\1 ${input_path}|" \
        -e "s|^\(output_dir:\).*|\1 ${output_dir}|" \
        "$CONFIG_TEMPLATE" > "$temp_config"

    if python App/main.py --task inference --config "$temp_config"; then
        n_done=$((n_done + 1))
        echo "Finished: $sample_name"
    else
        n_failed=$((n_failed + 1))
        echo "[FAIL] $sample_name (continuing)"
    fi
    echo ""
done

echo "============================================"
echo "Total: $n_total | Done: $n_done | Failed: $n_failed | Skipped: $n_skipped"
echo "All samples processed."
