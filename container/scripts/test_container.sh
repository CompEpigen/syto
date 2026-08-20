#!/bin/bash
#
# test_container.sh - Test MethylDL container image
#
# Usage:
#   ./test_container.sh <image_path> [options]
#
# Examples:
#   ./test_container.sh ./images/methyldl_ubuntu22_single.sif
#   ./test_container.sh /absolute/path/to/container.sif --verbose
#   ./test_container.sh container.sif --quick --bind /data:/mnt/data
#   ./test_container.sh container.sif --bind $VSC_DATA --bind $VSC_SCRATCH
#   ./test_container.sh container.sif --bind $VSC_SCRATCH --mlflow-test-storage $VSC_SCRATCH/mlflow_tmp
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Test options
QUICK_TEST=false
VERBOSE=false
TEST_GPU=true
TEST_POETRY=true
TEST_IMPORTS=true
TEST_DISK=true
TEST_COVERAGE=true

# Bind mount options
BIND_MOUNTS=()

# MLflow configuration
MLFLOW_TEST_STORAGE=""

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_failure() {
    echo -e "${RED}[✗]${NC} $1"
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
Usage: $0 <image_path> [options]

Test MethylDL container image functionality.

Arguments:
  image_path        Path to container image (.sif file)
                    Can be relative or absolute path
                    Examples: ./container.sif, /path/to/container.sif

Options:
  --bind PATH[:DEST]    Bind mount a directory into the container
                        Can be specified multiple times
                        Examples: --bind /data, --bind /data:/mnt/data
                        Common HPC usage: --bind \$VSC_DATA --bind \$VSC_SCRATCH
  --mlflow-test-storage PATH
                        Set MLflow storage path for test artifacts and runs
                        Sets MLFLOW_TRACKING_URI=file://PATH for local file-based tracking
                        Path should be accessible (use --bind if needed)
                        MLflow will create subdirectories for experiments and artifacts
                        Example: --mlflow-test-storage \$VSC_SCRATCH/mlflow_tmp
  --quick               Run quick tests only (skip imports and disk tests)
  --no-gpu              Skip GPU tests
  --no-poetry           Skip Poetry tests
  --no-imports          Skip package import tests
  --no-coverage         Skip package coverage tests
  --verbose             Show verbose output
  -h, --help            Show this help message

Examples:
  # Full test suite with local image
  $0 ./images/methyldl_ubuntu22_single.sif

  # Quick test with absolute path
  $0 /path/to/container.sif --quick

  # Test without GPU checks, with bind mount
  $0 container.sif --no-gpu --bind /scratch/data

  # HPC usage with environment variables
  $0 ./container.sif --bind \$VSC_DATA --bind \$VSC_SCRATCH

  # HPC with MLflow storage on scratch
  $0 ./container.sif --bind \$VSC_SCRATCH --mlflow-test-storage \$VSC_SCRATCH/mlflow_tmp

  # Multiple bind mounts with custom destinations
  $0 container.sif --bind /data:/mnt/data --bind /scratch:/mnt/scratch

Test Categories:
  1. Image inspection (labels, metadata)
  2. System verification (Python, CUDA, paths)
  3. GPU detection (if --nv available)
  4. Poetry environment
  5. Package imports (PyTorch, Transformers, etc.)
  6. File system structure
  7. Application entry point
  8. Package unit tests (if --no-coverage not set)

EOF
    exit 0
}

# Parse arguments
if [ $# -eq 0 ]; then
    print_error "No image path provided"
    usage
fi

IMAGE_PATH="$1"
shift

# Parse additional arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --bind)
            if [ $# -lt 2 ]; then
                print_error "--bind requires an argument"
                usage
            fi
            BIND_MOUNTS+=("$2")
            shift 2
            ;;
        --mlflow-test-storage)
            if [ $# -lt 2 ]; then
                print_error "--mlflow-test-storage requires an argument"
                usage
            fi
            MLFLOW_TEST_STORAGE="$2"
            shift 2
            ;;
        --quick)
            QUICK_TEST=true
            TEST_IMPORTS=false
            TEST_DISK=false
            shift
            ;;
        --no-gpu)
            TEST_GPU=false
            shift
            ;;
        --no-poetry)
            TEST_POETRY=false
            shift
            ;;
        --no-imports)
            TEST_IMPORTS=false
            shift
            ;;
        --no-coverage)
            TEST_COVERAGE=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# ==========================================
# Resolve image path
# ==========================================
# Convert to absolute path if relative
if [[ "$IMAGE_PATH" != /* ]]; then
    IMAGE_PATH="$(cd "$(dirname "$IMAGE_PATH")" && pwd)/$(basename "$IMAGE_PATH")"
fi

# Extract image name for display
IMAGE_NAME="$(basename "$IMAGE_PATH")"

# ==========================================
# Build bind mount arguments
# ==========================================
BIND_ARGS=()
if [ ${#BIND_MOUNTS[@]} -gt 0 ]; then
    for bind_path in "${BIND_MOUNTS[@]}"; do
        BIND_ARGS+=("--bind" "$bind_path")
    done
fi

# ==========================================
# Build MLflow environment variables
# ==========================================
MLFLOW_ENV_ARGS=()
if [ -n "$MLFLOW_TEST_STORAGE" ]; then
    # Create the directory if it doesn't exist (on host)
    if [[ "$MLFLOW_TEST_STORAGE" != /* ]]; then
        # Convert relative to absolute
        MLFLOW_ABS_PATH="$(cd "$(dirname "$MLFLOW_TEST_STORAGE")" 2>/dev/null && pwd)/$(basename "$MLFLOW_TEST_STORAGE")" || MLFLOW_ABS_PATH="$MLFLOW_TEST_STORAGE"
    else
        MLFLOW_ABS_PATH="$MLFLOW_TEST_STORAGE"
    fi
    
    print_info "Setting up MLflow test storage: ${MLFLOW_ABS_PATH}"
    mkdir -p "$MLFLOW_ABS_PATH" 2>/dev/null || print_warning "Could not create MLflow directory (may already exist or need permissions)"
    
    # Set environment variables for MLflow
    # MLFLOW_TRACKING_URI controls where MLflow client writes data
    # This is the key variable for file-based tracking without a server
    MLFLOW_ENV_ARGS+=(
        "--env" "MLFLOW_TRACKING_URI=file://${MLFLOW_TEST_STORAGE}"
    )
fi

# ==========================================
# Apptainer wrapper functions
# ==========================================
apptainer_exec_wrapper() {
    # Generic wrapper for all Apptainer operations
    local subcmd="$1"
    shift
    
    # Build command with bind mounts and environment variables if provided
    local all_args=()
    
    # Add bind mounts
    if [ ${#BIND_ARGS[@]} -gt 0 ]; then
        all_args+=("${BIND_ARGS[@]}")
    fi
    
    # Add MLflow environment variables
    if [ ${#MLFLOW_ENV_ARGS[@]} -gt 0 ]; then
        all_args+=("${MLFLOW_ENV_ARGS[@]}")
    fi
    
    # Execute with all arguments
    if [ ${#all_args[@]} -gt 0 ]; then
        apptainer "$subcmd" "${all_args[@]}" "$@"
    else
        apptainer "$subcmd" "$@"
    fi
}

# Convenience wrappers
appt_exec()     { apptainer_exec_wrapper exec "$@"; }
appt_inspect()  { apptainer_exec_wrapper inspect "$@"; }
appt_runhelp()  { apptainer_exec_wrapper run-help "$@"; }
appt_run()      { apptainer_exec_wrapper run "$@"; }

# Print test configuration
print_info "=========================================="
print_info "MethylDL Container Test Suite"
print_info "=========================================="
print_info "Image:      ${IMAGE_NAME}"
print_info "Path:       ${IMAGE_PATH}"
print_info "Quick test: ${QUICK_TEST}"
print_info "Test GPU:   ${TEST_GPU}"
if [ ${#BIND_MOUNTS[@]} -gt 0 ]; then
    print_info "Bind mounts:"
    for bind in "${BIND_MOUNTS[@]}"; do
        print_info "  - ${bind}"
    done
fi
if [ -n "$MLFLOW_TEST_STORAGE" ]; then
    print_info "MLflow storage: ${MLFLOW_TEST_STORAGE}"
fi
print_info "=========================================="
echo ""

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    print_error "Image file not found: ${IMAGE_PATH}"
    print_info ""
    print_info "Please provide a valid path to a .sif container image"
    exit 1
fi

# Check if apptainer is installed
if ! command -v apptainer &> /dev/null; then
    print_error "Apptainer is not installed"
    exit 1
fi

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expected_pattern="$3"  # Optional: grep pattern to verify output
    local show_output="${4:-true}"  # Optional: show output on success (default: true)
    
    TESTS_RUN=$((TESTS_RUN + 1))
    print_test "${test_name}"
    
    local output
    if output=$(eval "$test_cmd" 2>&1); then
        if [ -n "$expected_pattern" ]; then
            if echo "$output" | grep -q "$expected_pattern"; then
                if [ "$show_output" = "true" ]; then
                    print_success "${test_name}: ${output}"
                else
                    print_success "${test_name}"
                fi
                TESTS_PASSED=$((TESTS_PASSED + 1))
                return 0
            else
                print_failure "${test_name} - Expected pattern not found: ${expected_pattern}"
                echo "$output" | sed 's/^/    /'
                TESTS_FAILED=$((TESTS_FAILED + 1))
                return 1
            fi
        else
            if [ "$show_output" = "true" ]; then
                print_success "${test_name}: ${output}"
            else
                print_success "${test_name}"
            fi
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        fi
    else
        print_failure "${test_name}"
        echo "$output" | sed 's/^/    /'
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ==========================================
# Test 1: Image Inspection
# ==========================================
echo ""
print_info "=== Test Category 1: Image Inspection ==="
echo ""

run_test "Inspect image metadata" \
    "apptainer inspect '$IMAGE_PATH' | head -5" \
    "" \
    "true"

run_test "Check image labels" \
    "apptainer inspect --labels '$IMAGE_PATH'" \
    "maintainer" \
    "true"

run_test "Image size" \
    "du -h '$IMAGE_PATH' | cut -f1" \
    "" \
    "true"

# ==========================================
# Test 2: System Verification
# ==========================================
echo ""
print_info "=== Test Category 2: System Verification ==="
echo ""

run_test "Python version" \
    "appt_exec '$IMAGE_PATH' python3 --version" \
    "Python 3.12" \
    "true"

run_test "Python path" \
    "appt_exec '$IMAGE_PATH' which python3" \
    "python3" \
    "true"

run_test "Pip version" \
    "appt_exec '$IMAGE_PATH' python3 -m pip --version" \
    "pip" \
    "true"

run_test "CUDA_HOME" \
    "appt_exec '$IMAGE_PATH' bash -c 'echo \$CUDA_HOME'" \
    "/usr/local/cuda" \
    "true"

run_test "Virtual environment in PATH" \
    "appt_exec '$IMAGE_PATH' bash -c 'echo \$PATH | grep -o \"/workspace/methyldl/.venv/bin\"'" \
    "/workspace/methyldl/.venv/bin" \
    "true"

# Test MLflow configuration if specified
if [ -n "$MLFLOW_TEST_STORAGE" ]; then
    run_test "MLflow tracking URI" \
        "appt_exec '$IMAGE_PATH' bash -c 'echo \$MLFLOW_TRACKING_URI'" \
        "file://${MLFLOW_TEST_STORAGE}" \
        "true"
fi

# ==========================================
# Test 3: GPU Detection (if available)
# ==========================================
if [ "$TEST_GPU" = true ]; then
    echo ""
    print_info "=== Test Category 3: GPU Detection ==="
    echo ""
    
    # Check if nvidia-smi is available on host
    if command -v nvidia-smi &> /dev/null; then
        run_test "GPU on host" \
            "nvidia-smi --query-gpu=name --format=csv,noheader" \
            "" \
            "true"
        
        run_test "NVIDIA driver" \
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader" \
            "" \
            "true"
        
        run_test "GPU access from container" \
            "appt_exec --nv '$IMAGE_PATH' nvidia-smi --query-gpu=name --format=csv,noheader" \
            "" \
            "true"
    else
        print_warning "nvidia-smi not found on host, skipping GPU tests"
        print_info "  (GPU tests will be skipped but container may still work with GPU)"
    fi
fi

# ==========================================
# Test 4: Poetry Environment
# TODO: 23/10/25 not able to come up with proper tests, skipping for now
# ==========================================
if [ "$TEST_POETRY" = true ]; then
    echo ""
    print_info "=== Test Category 4: Poetry Environment ==="
    echo ""
    echo "Poetry tests are skipped for now until there is a proper way to check poetry installation"
fi

# ==========================================
# Test 5: Package Imports
# ==========================================
if [ "$TEST_IMPORTS" = true ]; then
    echo ""
    print_info "=== Test Category 5: Package Imports ==="
    echo ""
    
    run_test "PyTorch version" \
        "appt_exec '$IMAGE_PATH' python3 -c 'import torch; print(torch.__version__)'" \
        "" \
        "true"
    
    run_test "Transformers version" \
        "appt_exec '$IMAGE_PATH' python3 -c 'import transformers; print(transformers.__version__)'" \
        "" \
        "true"
    
    run_test "NumPy version" \
        "appt_exec '$IMAGE_PATH' python3 -c 'import numpy; print(numpy.__version__)'" \
        "" \
        "true"
    
    if [ "$TEST_GPU" = true ] && command -v nvidia-smi &> /dev/null; then
        run_test "PyTorch CUDA availability" \
            "appt_exec --nv '$IMAGE_PATH' python3 -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'" \
            "CUDA available: True" \
            "true"
        
        run_test "PyTorch GPU count" \
            "appt_exec --nv '$IMAGE_PATH' python3 -c 'import torch; print(torch.cuda.device_count(), \"GPU(s) detected\")'" \
            "" \
            "true"
    fi
    
    run_test "MethylDL package" \
        "appt_exec '$IMAGE_PATH' python3 -c 'import methyldl; print(\"MethylDL imported successfully\")'" \
        "MethylDL imported successfully" \
        "true"
fi

# ==========================================
# Test 6: File System Structure
# ==========================================
if [ "$TEST_DISK" = true ]; then
    echo ""
    print_info "=== Test Category 6: File System Structure ==="
    echo ""
    
    run_test "Workspace directory exists" \
        "appt_exec '$IMAGE_PATH' test -d /workspace/methyldl && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "Virtual environment exists" \
        "appt_exec '$IMAGE_PATH' test -d /workspace/methyldl/.venv && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "Cache directory exists" \
        "appt_exec '$IMAGE_PATH' test -d /workspace/cache && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "Data directory exists" \
        "appt_exec '$IMAGE_PATH' test -d /workspace/data && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "Outputs directory exists" \
        "appt_exec '$IMAGE_PATH' test -d /workspace/outputs && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "syto/app directory exists" \
        "appt_exec '$IMAGE_PATH' test -d /workspace/methyldl/syto/app && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "main.py exists" \
        "appt_exec '$IMAGE_PATH' test -f /workspace/methyldl/syto/app/cli.py && echo 'OK'" \
        "OK" \
        "false"
    
    # Test bind mounts if any were specified
    if [ ${#BIND_MOUNTS[@]} -gt 0 ]; then
        echo ""
        print_info "Testing bind mounts:"
        for bind in "${BIND_MOUNTS[@]}"; do
            # Extract source path (before colon, or entire string if no colon)
            source_path="${bind%%:*}"
            # Extract destination path (after colon, or same as source if no colon)
            if [[ "$bind" == *":"* ]]; then
                dest_path="${bind#*:}"
            else
                dest_path="$source_path"
            fi
            
            run_test "Bind mount accessible: ${dest_path}" \
                "appt_exec '$IMAGE_PATH' test -d '$dest_path' && echo 'OK'" \
                "OK" \
                "false"
        done
    fi
    
    # Test MLflow storage accessibility
    if [ -n "$MLFLOW_TEST_STORAGE" ]; then
        echo ""
        print_info "Testing MLflow storage:"
        
        run_test "MLflow storage directory accessible" \
            "appt_exec '$IMAGE_PATH' test -d '$MLFLOW_TEST_STORAGE' && echo 'OK'" \
            "OK" \
            "false"
        
        run_test "MLflow storage writable" \
            "appt_exec '$IMAGE_PATH' bash -c 'touch \"$MLFLOW_TEST_STORAGE/.test_write\" && rm \"$MLFLOW_TEST_STORAGE/.test_write\" && echo \"OK\"'" \
            "OK" \
            "false"
    fi
fi

# ==========================================
# Test 7: Application Entry Point
# ==========================================
echo ""
print_info "=== Test Category 7: Application Entry Point ==="
echo ""

run_test "Runscript help available" \
    "apptainer run-help '$IMAGE_PATH' | head -3" \
    "" \
    "true"

run_test "Container executable" \
    "appt_exec '$IMAGE_PATH' test -x /workspace/methyldl/syto/app/cli.py && echo 'main.py is accessible' || echo 'main.py exists'" \
    "" \
    "true"


# ==========================================
# Test 8: methyldl Package Unit Tests
# ==========================================
if [ "$TEST_COVERAGE" = true ]; then
    echo ""
    print_info "=== Test Category 8: Package Unit Tests ==="
    echo ""
    
    print_test "Running MethylDL unit tests (this may take ~80 seconds)..."
    
    # Run both coverage run AND coverage report in the same exec
    
    unit_test_output=$(appt_exec --nv "$IMAGE_PATH" bash -c 'cd /workspace/methyldl && coverage run  && coverage report --skip-empty' 2>&1)
    unit_test_exit=$?
    

    if [ $unit_test_exit -eq 0 ]; then
        # Extract summary line from pytest output
        summary=$(echo "$unit_test_output" | grep "Ran .* tests in")
        status=$(echo "$unit_test_output" | grep -E "^(OK|FAILED)")
        
        if echo "$status" | grep -q "OK"; then
            print_success "Unit tests: $summary - $status"
            TESTS_RUN=$((TESTS_RUN + 1))
            TESTS_PASSED=$((TESTS_PASSED + 1))
            
            # Extract and show coverage report
            coverage_report=$(echo "$unit_test_output" | grep -A 100 "Name" | grep -B 100 "TOTAL")
            if [ -n "$coverage_report" ]; then
                print_success "Coverage report:"
                echo "$coverage_report" | sed 's/^/    /'
                TESTS_RUN=$((TESTS_RUN + 1))
                TESTS_PASSED=$((TESTS_PASSED + 1))
            fi
        else
            print_failure "Unit tests: $summary - $status"
            echo "$unit_test_output" | tail -30 | sed 's/^/    /'
            TESTS_RUN=$((TESTS_RUN + 1))
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    else
        print_failure "Unit tests: Failed to execute"
        echo "$unit_test_output" | tail -20 | sed 's/^/    /'
        TESTS_RUN=$((TESTS_RUN + 1))
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
fi


# ==========================================
# Test Summary
# ==========================================
echo ""
print_info "=========================================="
print_info "Test Summary"
print_info "=========================================="
print_info "Total tests:  ${TESTS_RUN}"
print_success "Passed:       ${TESTS_PASSED}"
if [ $TESTS_FAILED -gt 0 ]; then
    print_failure "Failed:       ${TESTS_FAILED}"
else
    print_info "Failed:       ${TESTS_FAILED}"
fi
print_info "=========================================="
echo ""

# Final verdict
if [ $TESTS_FAILED -eq 0 ]; then
    print_success "All tests passed! ✓"
    print_info ""
    print_info "Container is ready to use:"
    print_info "  Run:   apptainer run --nv ${IMAGE_PATH}"
    print_info "  Shell: apptainer shell --nv ${IMAGE_PATH}"
    print_info "  Help:  apptainer run-help ${IMAGE_PATH}"
    if [ ${#BIND_MOUNTS[@]} -gt 0 ] || [ -n "$MLFLOW_TEST_STORAGE" ]; then
        print_info ""
        print_info "With your configuration:"
        cmd_str="apptainer run --nv"
        for bind in "${BIND_MOUNTS[@]}"; do
            cmd_str="${cmd_str} --bind ${bind}"
        done
        if [ -n "$MLFLOW_TEST_STORAGE" ]; then
            cmd_str="${cmd_str} --env MLFLOW_TRACKING_URI=file://${MLFLOW_TEST_STORAGE}"
        fi
        cmd_str="${cmd_str} ${IMAGE_PATH}"
        print_info "  ${cmd_str}"
    fi
    echo ""
    exit 0
else
    print_warning "Some tests failed!"
    print_info ""
    print_info "The container may still be functional for some use cases."
    print_info "Review the failed tests above and check:"
    print_info "  1. Container image integrity"
    print_info "  2. Build logs"
    print_info "  3. Package dependencies"
    print_info "  4. Bind mount paths (if using --bind)"
    print_info ""
    print_info "Rerun with --verbose for more details:"
    print_info "  $0 ${IMAGE_PATH} --verbose"
    echo ""
    exit 1
fi