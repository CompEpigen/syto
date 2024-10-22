# Define the paths
$genomeScriptPath = "./methyldl/data/genome.py"
$referenceGenomesFolder = "./Data/Raw/ReferenceGenomes"
$outputFolder = "./Data/Refined/ReferenceGenomes"

$curtimestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logFilePath = "./Logs/refine_reference_genome_log_${curtimestamp}.txt "

# Define the kmer and length parameters
$kmers = @(1, 3)
$lengths = @(510, 10000)

# Ensure the output directory exists
if (-not (Test-Path $outputFolder)) {
    New-Item -Path $outputFolder -ItemType Directory
}

# Function to log messages with a timestamp
function Write-Log {
    param (
        [string]$message
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "${timestamp}: $message"
    
    # Print to console
    Write-Host $logMessage
    
    # Write to log file
    Add-Content -Path $logFilePath -Value $logMessage
}

# Start logging
$startTime = Get-Date
Write-Log "Processing started."

# Get all the reference genome files in the folder
$genomeFiles = Get-ChildItem -Path $referenceGenomesFolder -File

# Loop through each genome file
foreach ($genomeFile in $genomeFiles) {
    $filePath = $genomeFile.FullName

    # Loop through each kmer value
    foreach ($kmer in $kmers) {
        
        # Loop through each length value
        foreach ($length in $lengths) {
            # Construct the output path
            $outputFilePath = "$outputFolder/$($genomeFile.BaseName)_kmer${kmer}_seqlen_${length}.txt"
            
            # Log the current process
            Write-Log "Processing file: $genomeFile with kmer: $kmer and length: $length."
            
            # Execute the genome.py script with the given parameters
            python $genomeScriptPath --file_path $filePath --kmer $kmer --length $length --output_path $outputFilePath
            
            # Log completion of the current task
            Write-Log "Completed file: $genomeFile with kmer: $kmer and length: $length. Output saved to $outputFilePath."
        }
    }
}

# End logging
$endTime = Get-Date
$elapsedTime = $endTime - $startTime
Write-Log "Processing completed. Total time: $($elapsedTime.Hours) hours, $($elapsedTime.Minutes) minutes, $($elapsedTime.Seconds) seconds."