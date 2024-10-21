#!/bin/bash
# Save in: ./Scripts/bash/fetch_reference_genomes.sh

# Set the target directory
TARGET_DIR="../../Data/Raw/ReferenceGenomes"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Define URLs
URLS=(
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz"
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz"
)

# Download and extract each file
for url in "${URLS[@]}"; do
    file_name=$(basename "$url")
    destination_path="$TARGET_DIR/$file_name"
    wget -O "$destination_path" "$url"
done

echo "Download complete."
