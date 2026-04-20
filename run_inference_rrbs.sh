#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch inference runner for MethylBERT
# Loops over all files in INPUT_DIR, creates a per-file config
# with the correct input path and output directory, then runs
# the inference pipeline.
# ============================================================

INPUT_DIR="/data/dmytro/cfSortData/RecoveredReads"
CONFIG_TEMPLATE="App/config/rrbs/inference.yaml"
TEMP_CONFIG_DIR="App/config/rrbs/tmp_configs"

mkdir -p "$TEMP_CONFIG_DIR"

for filepath in "$INPUT_DIR"/*; do
    [ -f "$filepath" ] || continue

    filename=$(basename "$filepath")
    sample_name="${filename%.*}"          # strip extension
    sample_name="${sample_name/_reads/}"  # remove _reads
    output_dir="/home/luna.kuleuven.be/u0169940/Repos/methyldl/Experiments/RRBS/methylBert_Loyfer_205files_U25_attentionClassifier_hg38_dmr_ctype_label_mincpg_4_minlen_10_d041_postfiltered_min_length_50_hard_labels_minibatch_balanced/pseudobulk/deconvolutions/${sample_name}"
    temp_config="${TEMP_CONFIG_DIR}/inference_${sample_name}.yaml"

    # # Skip if output directory already exists and contains predictions
    # if [ -d "$output_dir" ] && [ -f "$output_dir/predictions.pkl" ]; then
    #     echo "[SKIP] Already processed: $filename ($output_dir/predictions.pkl exists)"
    #     continue
    # fi

    echo "============================================"
    echo "Processing: $filename"
    echo "Output dir: $output_dir"
    echo "============================================"

    # Create a per-sample config by replacing the relevant fields
    sed \
        -e "s|^\(\s*parsed_reads_path:\).*|\1 ${filepath}|" \
        -e "s|^\(\s*predicted_reads_path:\).*|\1 ${output_dir}/predictions.pkl|" \
        -e "s|^\(\s*output_dir:\).*|\1 ${output_dir}|" \
        "$CONFIG_TEMPLATE" > "$temp_config"

    # Run inference
    python App/main.py --task inference --config "$temp_config"

    echo "Finished: $filename"
    echo ""
done

echo "All samples processed."