#!/bin/bash
#
# build_container.sh - Build MethylDL container from HPCCM recipe
#
# Usage:
#   ./build_container.sh <recipe_name> [options]
#
# Examples:
#   ./build_container.sh methyldl_ubuntu22_single
#   ./build_container.sh methyldl_rockylinux9_multi --fakeroot
#   ./build_container.sh methyldl_ubuntu22_single --sudo --clean
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$CONTAINER_DIR")"

# Default directories
RECIPE_DIR="${CONTAINER_DIR}/hpccm"
DEF_DIR="${CONTAINER_DIR}/definitions"
IMAGE_DIR="${CONTAINER_DIR}/images"

# Default options
USE_SUDO=false
USE_FAKEROOT=false
CLEAN_BUILD=false
SKIP_GENERATE=false
VERBOSE=false
OUTPUT_DIR=""

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 <recipe_name> [options]

Build MethylDL container from HPCCM recipe.

Arguments:
  recipe_name       Name of the recipe (without .py extension)
                    Examples: methyldl_ubuntu22_single, methyldl_rockylinux9_multi

Options:
  --sudo            Use sudo for apptainer build (default: no)
  --fakeroot        Use fakeroot for apptainer build (default: no)
  --output-dir DIR  Put Apptainer config/cache/tmp files under DIR
  --clean           Remove existing definition and image files before building
  --skip-generate   Skip HPCCM generation, use existing .def file
  --verbose         Show verbose output
  -h, --help        Show this help message

Examples:
  # Build with fakeroot (no sudo required)
  $0 methyldl_ubuntu22_single --fakeroot

  # Build with sudo
  $0 methyldl_ubuntu22_single --sudo

  # Build with Apptainer cache/config/tmp stored outside home
  $0 methyldl_ubuntu22_single --fakeroot --output-dir /path/to/build-output

  # Clean build with sudo
  $0 methyldl_rockylinux9_multi --sudo --clean

  # Use existing definition file, just rebuild image
  $0 methyldl_ubuntu22_single --sudo --skip-generate

Directory structure:
  Recipes:      ${RECIPE_DIR}
  Definitions:  ${DEF_DIR}
  Images:       ${IMAGE_DIR}
EOF
    exit 0
}

# Parse arguments
if [ $# -eq 0 ]; then
    print_error "No recipe name provided"
    usage
fi

RECIPE_NAME="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --sudo)
            USE_SUDO=true
            ;;
        --fakeroot)
            USE_FAKEROOT=true
            ;;
        --output-dir)
            if [ $# -lt 2 ]; then
                print_error "--output-dir requires an argument"
                usage
            fi
            OUTPUT_DIR="$2"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            ;;
        --skip-generate)
            SKIP_GENERATE=true
            ;;
        --verbose)
            VERBOSE=true
            set -x  # Enable shell command tracing
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
    shift
done

# Check if both sudo and fakeroot are specified
if [ "$USE_SUDO" = true ] && [ "$USE_FAKEROOT" = true ]; then
    print_error "Cannot use both --sudo and --fakeroot. Choose one."
    exit 1
fi

# Set default to fakeroot if neither is specified
if [ "$USE_SUDO" = false ] && [ "$USE_FAKEROOT" = false ]; then
    print_warning "No build method specified, defaulting to --fakeroot"
    USE_FAKEROOT=true
fi

# Define file paths
RECIPE_FILE="${RECIPE_DIR}/${RECIPE_NAME}.py"
DEF_FILE="${DEF_DIR}/${RECIPE_NAME}.def"
IMAGE_FILE="${IMAGE_DIR}/${RECIPE_NAME}.sif"

# Optional Apptainer runtime directories
APPTAINER_CONFIG_DIR=""
APPTAINER_CACHE_DIR=""
APPTAINER_TMP_DIR=""

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

    APPTAINER_CONFIG_DIR="${OUTPUT_DIR}/apptainer-config"
    APPTAINER_CACHE_DIR="${OUTPUT_DIR}/apptainer-cache"
    APPTAINER_TMP_DIR="${OUTPUT_DIR}/apptainer-tmp"

    mkdir -p "$APPTAINER_CONFIG_DIR" "$APPTAINER_CACHE_DIR" "$APPTAINER_TMP_DIR"

    export APPTAINER_CONFIGDIR="$APPTAINER_CONFIG_DIR"
    export APPTAINER_CACHEDIR="$APPTAINER_CACHE_DIR"
    export APPTAINER_TMPDIR="$APPTAINER_TMP_DIR"
    export TMPDIR="$APPTAINER_TMP_DIR"
fi

# Print build configuration
print_info "=========================================="
print_info "MethylDL Container Build Script"
print_info "=========================================="
print_info "Recipe:     ${RECIPE_NAME}"
print_info "Recipe dir: ${RECIPE_DIR}"
print_info "Def dir:    ${DEF_DIR}"
print_info "Image dir:  ${IMAGE_DIR}"
if [ -n "$OUTPUT_DIR" ]; then
    print_info "Output dir: ${OUTPUT_DIR}"
    print_info "Apptainer config dir: ${APPTAINER_CONFIG_DIR}"
    print_info "Apptainer cache dir:  ${APPTAINER_CACHE_DIR}"
    print_info "Apptainer tmp dir:    ${APPTAINER_TMP_DIR}"
fi
print_info "Build mode: $([ "$USE_SUDO" = true ] && echo "sudo" || echo "fakeroot")"
print_info "Clean:      ${CLEAN_BUILD}"
print_info "=========================================="

# Check if HPCCM is installed
if ! command -v hpccm &> /dev/null; then
    print_error "HPCCM is not installed"
    print_info "Install with: pip install hpccm"
    exit 1
fi

# Check if apptainer is installed
if ! command -v apptainer &> /dev/null; then
    print_error "Apptainer is not installed"
    print_info "Install from: https://apptainer.org/docs/admin/main/installation.html"
    exit 1
fi

# Check if recipe file exists
if [ ! -f "$RECIPE_FILE" ]; then
    print_error "Recipe file not found: ${RECIPE_FILE}"
    print_info "Available recipes:"
    ls -1 "${RECIPE_DIR}"/*.py 2>/dev/null | xargs -n 1 basename | sed 's/^/  - /' || echo "  (none)"
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$DEF_DIR"
mkdir -p "$IMAGE_DIR"

# Clean old files if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_info "Cleaning old files..."
    [ -f "$DEF_FILE" ] && rm -f "$DEF_FILE" && print_info "  Removed: ${DEF_FILE}"
    [ -f "$IMAGE_FILE" ] && rm -f "$IMAGE_FILE" && print_info "  Removed: ${IMAGE_FILE}"
fi

# Generate definition file
if [ "$SKIP_GENERATE" = false ]; then
    print_info "Generating Singularity definition file..."
    print_info "  Command: hpccm --recipe ${RECIPE_FILE} --format singularity"
    
    if hpccm --recipe "$RECIPE_FILE" --format singularity > "$DEF_FILE"; then
        print_success "Definition file generated: ${DEF_FILE}"
        print_info "  Size: $(du -h "$DEF_FILE" | cut -f1)"
        print_info "  Lines: $(wc -l < "$DEF_FILE")"
    else
        print_error "Failed to generate definition file"
        exit 1
    fi
else
    print_warning "Skipping HPCCM generation, using existing definition file"
    if [ ! -f "$DEF_FILE" ]; then
        print_error "Definition file not found: ${DEF_FILE}"
        exit 1
    fi
fi

# Build container image
print_info "Building Singularity image..."

# Construct build command
BUILD_CMD="apptainer build"
if [ "$USE_SUDO" = true ]; then
    BUILD_CMD="sudo --preserve-env=APPTAINER_CONFIGDIR,APPTAINER_CACHEDIR,APPTAINER_TMPDIR,TMPDIR ${BUILD_CMD}"
elif [ "$USE_FAKEROOT" = true ]; then
    BUILD_CMD="${BUILD_CMD} --fakeroot"
fi
BUILD_CMD="${BUILD_CMD} ${IMAGE_FILE} ${DEF_FILE}"

print_info "  Command: ${BUILD_CMD}"
print_info "  Working directory: ${PROJECT_ROOT}"
print_warning "This may take 15-30 minutes depending on your system..."

# Record start time
START_TIME=$(date +%s)

# Change to project root before building
cd "$PROJECT_ROOT"

# Run build command
if eval "$BUILD_CMD"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))
    
    print_success "=========================================="
    print_success "Build completed successfully!"
    print_success "=========================================="
    print_info "Image file: ${IMAGE_FILE}"
    print_info "Image size: $(du -h "$IMAGE_FILE" | cut -f1)"
    print_info "Build time: ${DURATION_MIN}m ${DURATION_SEC}s"
    print_success "=========================================="
    print_info ""
    print_info "Next steps:"
    print_info "  1. Test the image: ./test_container.sh ${RECIPE_NAME}"
    print_info "  2. Run the container: apptainer run --nv ${IMAGE_FILE}"
    print_info "  3. Interactive shell: apptainer shell --nv ${IMAGE_FILE}"
    print_info ""
else
    print_error "Build failed!"
    print_info "Check the output above for errors"
    print_info "Common issues:"
    print_info "  - Missing dependencies in recipe"
    print_info "  - Network connectivity problems"
    print_info "  - Insufficient disk space"
    print_info "  - Permission issues (try --sudo or --fakeroot)"
    exit 1
fi

# Verify the image
print_info "Verifying image..."
if apptainer inspect "$IMAGE_FILE" &> /dev/null; then
    print_success "Image verification passed"
    print_info ""
    print_info "Image labels:"
    apptainer inspect --labels "$IMAGE_FILE" | sed 's/^/  /'
else
    print_warning "Image verification failed, but image may still be usable"
fi

print_info ""
print_success "All done! 🎉"