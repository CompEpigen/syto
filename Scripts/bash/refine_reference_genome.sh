#!/bin/bash

# Define the paths
genomeScriptPath="./methyldl/data/genome.py"
referenceGenomesFolder="./Data/Raw/ReferenceGenomes"
outputFolder="./Data/Refined/ReferenceGenomes"

curtimestamp=$(date +"%Y-%m-%d_%H-%M-%S")
logFilePath="./Logs/refine_reference_genome_log_${curtimestamp}.txt"

# Define the kmer and length parameters
kmers=(1 3)
lengths=(510 10000)

# Ensure the output directory exists
if [ ! -d "$outputFolder" ]; then
    mkdir -p "$outputFolder"
fi

# Function to log messages with a timestamp
WriteLog() {
    message=$1
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    logMessage="${timestamp}: $message"

    # Print to console
    echo "$logMessage"

    # Write to log file
    echo "$logMessage" >> "$logFilePath"
}

# Start logging
startTime=$(date +%s)
WriteLog "Processing started."

# Get all the reference genome files in the folder
genomeFiles=("$referenceGenomesFolder"/*)

# Loop through each genome file
for genomeFile in "${genomeFiles[@]}"; do
    filePath="$genomeFile"

    # Get the base name of the genome file (without path)
    baseName=$(basename "$genomeFile")

    # Loop through each kmer value
    for kmer in "${kmers[@]}"; do

        # Loop through each length value
        for length in "${lengths[@]}"; do
            # Construct the output path
            outputFilePath="$outputFolder/${baseName}_kmer${kmer}_seqlen_${length}.txt"

            # Log the current process
            WriteLog "Processing file: $baseName with kmer: $kmer and length: $length."

            # Execute the genome.py script with the given parameters
            python "$genomeScriptPath" --file_path "$filePath" --kmer "$kmer" --length "$length" --output_path "$outputFilePath"

            # Log completion of the current task
            WriteLog "Completed file: $baseName with kmer: $kmer and length: $length. Output saved to $outputFilePath."
        done
    done
done

# End logging
endTime=$(date +%s)
elapsedTime=$((endTime - startTime))
hours=$((elapsedTime / 3600))
minutes=$(( (elapsedTime % 3600) / 60 ))
seconds=$((elapsedTime % 60))

WriteLog "Processing completed. Total time: ${hours} hours, ${minutes} minutes, ${seconds} seconds."
