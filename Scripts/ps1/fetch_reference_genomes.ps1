# Set the target directory
$TargetDirectory = "../../Data/Raw/ReferenceGenomes"
if (-Not (Test-Path -Path $TargetDirectory)) {
    New-Item -ItemType Directory -Path $TargetDirectory
}

# Define URLs
$urls = @(
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz",
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz"
)

# Download and extract each file
foreach ($url in $urls) {
    $fileName = [System.IO.Path]::GetFileName($url)
    $destinationPath = Join-Path -Path $TargetDirectory -ChildPath $fileName
    Invoke-WebRequest -Uri $url -OutFile $destinationPath
}

Write-Host "Download complete."
