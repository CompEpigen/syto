#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch inference/deconvolution runner for MethylBERT
# Auto-detects input mode from the template's input.type field:
#   - parsed_reads:    iterates over CSVs in parsed_reads_path's parent dir
#   - predicted_reads: iterates over per-sample subdirs in the predictions
#                      parent dir (each containing predictions.pkl)
# ============================================================

CONFIG_TEMPLATE="config/rrbs/inference.yaml"
TEMP_CONFIG_DIR="config/rrbs/tmp_configs"

mkdir -p "$TEMP_CONFIG_DIR"

# ---- Read input.type from template -----------------------------------------
# Match the `type:` line that sits under `input:` (first one after `input:`).
INPUT_TYPE=$(awk '
    /^input:/ { in_input=1; next }
    in_input && /^[^[:space:]]/ { in_input=0 }
    in_input && /^[[:space:]]+type:/ {
        sub(/^[[:space:]]+type:[[:space:]]*/, "")
        sub(/[[:space:]]*(#.*)?$/, "")
        print
        exit
    }
' "$CONFIG_TEMPLATE")

if [ -z "$INPUT_TYPE" ]; then
    echo "ERROR: Could not read input.type from template ${CONFIG_TEMPLATE}"
    exit 1
fi

# ---- Read output base dir --------------------------------------------------
OUTPUT_BASE_DIR=$(grep -E "^\s*output_dir:" "$CONFIG_TEMPLATE" | head -n1 | sed -E 's/^\s*output_dir:\s*//; s/\s*$//')
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR%/}"

if [ -z "$OUTPUT_BASE_DIR" ]; then
    echo "ERROR: Could not read output_dir from template ${CONFIG_TEMPLATE}"
    exit 1
fi

# ---- Mode-specific setup ---------------------------------------------------
case "$INPUT_TYPE" in
    parsed_reads)
        TEMPLATE_PARSED_PATH=$(grep -E "^\s*parsed_reads_path:" "$CONFIG_TEMPLATE" | head -n1 | sed -E 's/^\s*parsed_reads_path:\s*//; s/\s*$//')
        if [ -z "$TEMPLATE_PARSED_PATH" ] || [ "$TEMPLATE_PARSED_PATH" = "null" ]; then
            echo "ERROR: input.type is parsed_reads but parsed_reads_path is empty/null"
            exit 1
        fi
        INPUT_DIR=$(dirname "$TEMPLATE_PARSED_PATH")
        ;;
    predicted_reads)
        TEMPLATE_PRED_PATH=$(grep -E "^\s*predicted_reads_path:" "$CONFIG_TEMPLATE" | head -n1 | sed -E 's/^\s*predicted_reads_path:\s*//; s/\s*$//')
        if [ -z "$TEMPLATE_PRED_PATH" ] || [ "$TEMPLATE_PRED_PATH" = "null" ]; then
            echo "ERROR: input.type is predicted_reads but predicted_reads_path is empty/null"
            exit 1
        fi
        # Layout: <PREDICTIONS_DIR>/<sample>/predictions.pkl
        INPUT_DIR=$(dirname "$(dirname "$TEMPLATE_PRED_PATH")")
        ;;
    *)
        echo "ERROR: Unsupported input.type '$INPUT_TYPE' (expected parsed_reads or predicted_reads)"
        exit 1
        ;;
esac

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input dir does not exist: $INPUT_DIR"
    exit 1
fi

echo "Input mode:  $INPUT_TYPE"
echo "Input dir:   $INPUT_DIR"
echo "Output base: $OUTPUT_BASE_DIR"
echo ""

# ---- Main loop -------------------------------------------------------------
n_total=0
n_done=0
n_failed=0
n_missing=0

if [ "$INPUT_TYPE" = "parsed_reads" ]; then
    iter_glob=("$INPUT_DIR"/*.csv)
else
    iter_glob=("$INPUT_DIR"/*/)
fi

for entry in "${iter_glob[@]}"; do
    n_total=$((n_total + 1))

    if [ "$INPUT_TYPE" = "parsed_reads" ]; then
        [ -f "$entry" ] || { n_total=$((n_total - 1)); continue; }
        filename=$(basename "$entry")
        sample_name="${filename%.*}"
        sample_name="${sample_name/_reads/}"
        input_path="$entry"
        sed_input_expr="s|^\(\s*parsed_reads_path:\).*|\1 ${input_path}|"
    else
        [ -d "$entry" ] || { n_total=$((n_total - 1)); continue; }
        sample_name=$(basename "$entry")
        input_path="${entry%/}/predictions.pkl"
        if [ ! -f "$input_path" ]; then
            echo "[MISSING] $sample_name (no predictions.pkl in $entry)"
            n_missing=$((n_missing + 1))
            continue
        fi
        sed_input_expr="s|^\(\s*predicted_reads_path:\).*|\1 ${input_path}|"
    fi

    output_dir="${OUTPUT_BASE_DIR}/${sample_name}"
    temp_config="${TEMP_CONFIG_DIR}/inference_${sample_name}.yaml"

    # Skip-if-done check (toggle as needed)
    # done_marker="$output_dir/predictions.pkl"            # for parsed_reads mode
    # done_marker="$output_dir/deconvolution_proportions.csv"  # for predicted_reads mode
    # if [ -f "$done_marker" ]; then
    #     echo "[SKIP] $sample_name"
    #     n_done=$((n_done + 1))
    #     continue
    # fi

    echo "============================================"
    echo "Processing: $sample_name"
    echo "  Input:      $input_path"
    echo "  Output dir: $output_dir"
    echo "============================================"

    sed \
        -e "$sed_input_expr" \
        -e "s|^\(\s*output_dir:\).*|\1 ${output_dir}|" \
        "$CONFIG_TEMPLATE" > "$temp_config"

    if syto inference --config "$temp_config"; then
        n_done=$((n_done + 1))
        echo "Finished: $sample_name"
    else
        n_failed=$((n_failed + 1))
        echo "[FAIL] $sample_name (continuing)"
    fi
    echo ""
done

echo "============================================"
echo "Mode: $INPUT_TYPE | Total: $n_total | Done: $n_done | Failed: $n_failed | Missing: $n_missing"
echo "All samples processed."