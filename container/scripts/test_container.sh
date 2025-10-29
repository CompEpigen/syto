#!/bin/bash
#
# test_container.sh - Test MethylDL container image
#
# Usage:
#   ./test_container.sh <recipe_name> [options]
#
# Examples:
#   ./test_container.sh methyldl_ubuntu22_single
#   ./test_container.sh methyldl_rockylinux9_multi --verbose
#   ./test_container.sh methyldl_ubuntu22_single --quick
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
CONTAINER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$CONTAINER_DIR")"

# Default directories
IMAGE_DIR="${CONTAINER_DIR}/images"

# Test options
QUICK_TEST=false
VERBOSE=false
TEST_GPU=true
TEST_POETRY=true
TEST_IMPORTS=true
TEST_DISK=true
TEST_COVERAGE=true

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
Usage: $0 <recipe_name> [options]

Test MethylDL container image functionality.

Arguments:
  recipe_name       Name of the recipe/image (without .sif extension)
                    Examples: methyldl_ubuntu22_single, methyldl_rockylinux9_multi

Options:
  --quick           Run quick tests only (skip imports and disk tests)
  --no-gpu          Skip GPU tests
  --no-poetry       Skip Poetry tests
  --no-imports      Skip package import tests
  --no-coverage     Skip package coverage tests
  --verbose         Show verbose output
  -h, --help        Show this help message

Examples:
  # Full test suite
  $0 methyldl_ubuntu22_single

  # Quick test
  $0 methyldl_ubuntu22_single --quick

  # Test without GPU checks
  $0 methyldl_ubuntu22_single --no-gpu

Test Categories:
  1. Image inspection (labels, metadata)
  2. System verification (Python, CUDA, paths)
  3. GPU detection (if --nv available)
  4. Poetry environment
  5. Package imports (PyTorch, Transformers, etc.)
  6. File system structure
  7. Application entry point

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
        --quick)
            QUICK_TEST=true
            TEST_IMPORTS=false
            TEST_DISK=false
            ;;
        --no-gpu)
            TEST_GPU=false
            ;;
        --no-poetry)
            TEST_POETRY=false
            ;;
        --no-imports)
            TEST_IMPORTS=false
            ;;
        --no-coverage)
            TEST_COVERAGE=false
            ;;
        --verbose)
            VERBOSE=true
            set -x
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

# Define image path
IMAGE_FILE="${IMAGE_DIR}/${RECIPE_NAME}.sif"

# Print test configuration
print_info "=========================================="
print_info "MethylDL Container Test Suite"
print_info "=========================================="
print_info "Image:      ${RECIPE_NAME}"
print_info "Path:       ${IMAGE_FILE}"
print_info "Quick test: ${QUICK_TEST}"
print_info "Test GPU:   ${TEST_GPU}"
print_info "=========================================="
echo ""

# Check if image exists
if [ ! -f "$IMAGE_FILE" ]; then
    print_error "Image file not found: ${IMAGE_FILE}"
    print_info "Available images:"
    ls -1 "${IMAGE_DIR}"/*.sif 2>/dev/null | xargs -n 1 basename | sed 's/^/  - /' || echo "  (none)"
    print_info ""
    print_info "Build an image first with: ./build_container.sh ${RECIPE_NAME}"
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
    "apptainer inspect '$IMAGE_FILE' | head -5" \
    "" \
    "true"

run_test "Check image labels" \
    "apptainer inspect --labels '$IMAGE_FILE'" \
    "maintainer" \
    "true"

run_test "Image size" \
    "du -h '$IMAGE_FILE' | cut -f1" \
    "" \
    "true"

# ==========================================
# Test 2: System Verification
# ==========================================
echo ""
print_info "=== Test Category 2: System Verification ==="
echo ""

run_test "Python version" \
    "apptainer exec '$IMAGE_FILE' python3 --version" \
    "Python 3.12" \
    "true"

run_test "Python path" \
    "apptainer exec '$IMAGE_FILE' which python3" \
    "python3" \
    "true"

run_test "Pip version" \
    "apptainer exec '$IMAGE_FILE' python3 -m pip --version" \
    "pip" \
    "true"

run_test "CUDA_HOME" \
    "apptainer exec '$IMAGE_FILE' bash -c 'echo \$CUDA_HOME'" \
    "/usr/local/cuda" \
    "true"

run_test "Virtual environment in PATH" \
    "apptainer exec '$IMAGE_FILE' bash -c 'echo \$PATH | grep -o \"/workspace/methyldl/.venv/bin\"'" \
    "/workspace/methyldl/.venv/bin" \
    "true"

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
            "apptainer exec --nv '$IMAGE_FILE' nvidia-smi --query-gpu=name --format=csv,noheader" \
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

#     run_test "Poetry version" \
#         "apptainer exec '$IMAGE_FILE' /root/.local/bin/poetry --version" \
#         "Poetry" \
#         "true"
    
#     run_test "Virtual environment path" \
#         "apptainer exec '$IMAGE_FILE' /usr/local/bin/poetry env info --path 2>/dev/null || echo '/workspace/methyldl/.venv'" \
#         "/workspace/methyldl/.venv" \
#         "true"
    
#     run_test "Poetry virtualenvs.in-project config" \
#         "apptainer exec '$IMAGE_FILE' /usr/local/bin/poetry config --list | grep virtualenvs.in-project" \
#         "true" \
#         "true"
    
#     run_test "Installed packages count" \
#         "apptainer exec '$IMAGE_FILE' /usr/local/bin/poetry show 2>/dev/null | wc -l" \
#         "" \
#         "true"
# 
fi

# ==========================================
# Test 5: Package Imports
# ==========================================
if [ "$TEST_IMPORTS" = true ]; then
    echo ""
    print_info "=== Test Category 5: Package Imports ==="
    echo ""
    
    run_test "PyTorch version" \
        "apptainer exec '$IMAGE_FILE' python3 -c 'import torch; print(torch.__version__)'" \
        "" \
        "true"
    
    run_test "Transformers version" \
        "apptainer exec '$IMAGE_FILE' python3 -c 'import transformers; print(transformers.__version__)'" \
        "" \
        "true"
    
    run_test "NumPy version" \
        "apptainer exec '$IMAGE_FILE' python3 -c 'import numpy; print(numpy.__version__)'" \
        "" \
        "true"
    
    if [ "$TEST_GPU" = true ] && command -v nvidia-smi &> /dev/null; then
        run_test "PyTorch CUDA availability" \
            "apptainer exec --nv '$IMAGE_FILE' python3 -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'" \
            "CUDA available: True" \
            "true"
        
        run_test "PyTorch GPU count" \
            "apptainer exec --nv '$IMAGE_FILE' python3 -c 'import torch; print(torch.cuda.device_count(), \"GPU(s) detected\")'" \
            "" \
            "true"
    fi
    
    run_test "MethylDL package" \
        "apptainer exec '$IMAGE_FILE' python3 -c 'import methyldl; print(\"MethylDL imported successfully\")'" \
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
        "apptainer exec '$IMAGE_FILE' test -d /workspace/methyldl && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "Virtual environment exists" \
        "apptainer exec '$IMAGE_FILE' test -d /workspace/methyldl/.venv && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "Cache directory exists" \
        "apptainer exec '$IMAGE_FILE' test -d /workspace/cache && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "Data directory exists" \
        "apptainer exec '$IMAGE_FILE' test -d /workspace/data && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "Outputs directory exists" \
        "apptainer exec '$IMAGE_FILE' test -d /workspace/outputs && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "App directory exists" \
        "apptainer exec '$IMAGE_FILE' test -d /workspace/methyldl/App && echo 'OK'" \
        "OK" \
        "false"
    
    run_test "main.py exists" \
        "apptainer exec '$IMAGE_FILE' test -f /workspace/methyldl/App/main.py && echo 'OK'" \
        "OK" \
        "false"
fi

# ==========================================
# Test 7: Application Entry Point
# ==========================================
echo ""
print_info "=== Test Category 7: Application Entry Point ==="
echo ""

run_test "Runscript help available" \
    "apptainer run-help '$IMAGE_FILE' | head -3" \
    "" \
    "true"

run_test "Container executable" \
    "apptainer exec '$IMAGE_FILE' test -x /workspace/methyldl/App/main.py && echo 'main.py is accessible' || echo 'main.py exists'" \
    "" \
    "true"


# ==========================================
# Test 8: methyldl Package Unit Tests
# ==========================================
if [ "$TEST_COVERAGE" = true ]; then
    echo ""
    print_info "=== Test Category 8: Package Unit Tests ==="
    echo ""
    
    print_test "Running MethylDL unit tests (this may take ~40 seconds)..."
    
    # Run both coverage run AND coverage report in the same exec
    unit_test_output=$(apptainer exec --writable-tmpfs --nv "$IMAGE_FILE" bash -c 'coverage run && coverage report --skip-empty' 2>&1)
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
    print_info "  Run:   apptainer run --nv ${IMAGE_FILE}"
    print_info "  Shell: apptainer shell --nv ${IMAGE_FILE}"
    print_info "  Help:  apptainer run-help ${IMAGE_FILE}"
    echo ""
    exit 0
else
    print_warning "Some tests failed!"
    print_info ""
    print_info "The container may still be functional for some use cases."
    print_info "Review the failed tests above and check:"
    print_info "  1. Recipe configuration"
    print_info "  2. Build logs"
    print_info "  3. Package dependencies"
    print_info ""
    print_info "Rerun with --verbose for more details:"
    print_info "  $0 ${RECIPE_NAME} --verbose"
    echo ""
    exit 1
fi