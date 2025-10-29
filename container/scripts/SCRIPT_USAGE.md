# Container Build and Test Scripts

Two shell scripts for streamlined container building and testing workflow.

## Scripts Overview

### 1. `build_container.sh` - Build containers from HPCCM recipes
- Generates Singularity definition files using HPCCM
- Builds container images with Apptainer
- Supports both sudo and fakeroot builds
- Provides detailed progress and timing information

### 2. `test_container.sh` - Comprehensive container testing
- Tests 7 categories of functionality
- Validates Python, CUDA, Poetry, and packages
- GPU detection and PyTorch CUDA tests
- Produces detailed test reports

## Prerequisites

```bash
# Install HPCCM
pip install hpccm

# Install Apptainer (formerly Singularity)
# See: https://apptainer.org/docs/admin/main/installation.html

# For GPU tests, ensure NVIDIA drivers are installed
nvidia-smi
```

## Directory Structure

The scripts expect the following project structure:

```
project/
├── container/
│   ├── hpccm/                    # HPCCM recipe files (*.py)
│   ├── definitions/              # Generated definition files (*.def)
│   ├── images/                   # Built container images (*.sif)
│   └── scripts/                  # Build and test scripts
│       ├── build_container.sh    # Build script
│       └── test_container.sh     # Test script
├── methyldl/                     # Python package
├── App/                          # Application code
└── pyproject.toml                # Poetry config
```

**Note:** Scripts automatically detect their location and set paths relative to `container/` directory.

## Usage

### Building a Container

#### Basic Usage (with fakeroot - no sudo required)
```bash
cd container/scripts
./build_container.sh methyldl_ubuntu22_single --fakeroot
```

#### Using sudo
```bash
cd container/scripts
./build_container.sh methyldl_ubuntu22_single --sudo
```

#### Clean Build
```bash
# Remove old definition and image, then rebuild
cd container/scripts
./build_container.sh methyldl_ubuntu22_single --sudo --clean
```

#### Rebuild Image Only (skip HPCCM generation)
```bash
# Use existing .def file, just rebuild the image
./build_container.sh methyldl_ubuntu22_single --sudo --skip-generate
```

#### Verbose Mode
```bash
./build_container.sh methyldl_ubuntu22_single --fakeroot --verbose
```

### Build Options

| Option | Description |
|--------|-------------|
| `--sudo` | Use sudo for apptainer build (requires root) |
| `--fakeroot` | Use fakeroot for apptainer build (no root needed) |
| `--clean` | Remove existing .def and .sif files before building |
| `--skip-generate` | Skip HPCCM step, use existing .def file |
| `--verbose` | Show detailed command output |
| `-h, --help` | Display help message |

**Note:** If neither `--sudo` nor `--fakeroot` is specified, defaults to `--fakeroot`.

### Testing a Container

#### Full Test Suite
```bash
cd container/scripts
./test_container.sh methyldl_ubuntu22_single
```

#### Quick Test
```bash
# Skip import and disk tests for faster results
cd container/scripts
./test_container.sh methyldl_ubuntu22_single --quick
```

#### Test Without GPU
```bash
# Skip GPU-related tests
cd container/scripts
./test_container.sh methyldl_ubuntu22_single --no-gpu
```

#### Verbose Testing
```bash
./test_container.sh methyldl_ubuntu22_single --verbose
```

### Test Options

| Option | Description |
|--------|-------------|
| `--quick` | Run quick tests only (skip imports and disk) |
| `--no-gpu` | Skip GPU detection tests |
| `--no-poetry` | Skip Poetry environment tests |
| `--no-imports` | Skip package import tests |
| `--verbose` | Show detailed test output |
| `-h, --help` | Display help message |

## Test Categories

The test script validates 7 categories:

1. **Image Inspection** - Metadata, labels, size
2. **System Verification** - Python version, CUDA paths, environment
3. **GPU Detection** - NVIDIA GPU access (requires `--nv` flag)
4. **Poetry Environment** - Virtual environment, configuration
5. **Package Imports** - PyTorch, Transformers, NumPy, MethylDL
6. **File System Structure** - Required directories and files
7. **Application Entry Point** - Runscript, help functionality

## Complete Workflow Example

### Build, Test, and Deploy

```bash
# Navigate to scripts directory
cd container/scripts

# 1. Build the container (takes 15-30 minutes)
./build_container.sh methyldl_ubuntu22_single --fakeroot

# 2. Run full test suite
./test_container.sh methyldl_ubuntu22_single

# 3. If tests pass, use the container
cd ../..  # Back to project root
apptainer run --nv \
  -B ./data:/workspace/data \
  -B ./outputs:/workspace/outputs \
  container/images/methyldl_ubuntu22_single.sif --config config.yaml

# 4. Or get an interactive shell
apptainer shell --nv container/images/methyldl_ubuntu22_single.sif
```

### Rapid Development Cycle

```bash
# Navigate to scripts directory
cd container/scripts

# 1. Edit recipe
vim ../hpccm/methyldl_ubuntu22_single.py

# 2. Quick rebuild (clean + build)
./build_container.sh methyldl_ubuntu22_single --sudo --clean

# 3. Quick test
./test_container.sh methyldl_ubuntu22_single --quick

# 4. If quick test passes, run full test
./test_container.sh methyldl_ubuntu22_single
```

## Output Examples

### Build Script Output

```
[INFO] ==========================================
[INFO] MethylDL Container Build Script
[INFO] ==========================================
[INFO] Recipe:     methyldl_ubuntu22_single
[INFO] Recipe dir: /path/to/container/hpccm
[INFO] Def dir:    /path/to/container/definitions
[INFO] Image dir:  /path/to/container/images
[INFO] Build mode: fakeroot
[INFO] Clean:      false
[INFO] ==========================================
[INFO] Generating Singularity definition file...
[SUCCESS] Definition file generated: methyldl_ubuntu22_single.def
[INFO] Building Singularity image...
[WARNING] This may take 15-30 minutes depending on your system...
[SUCCESS] ==========================================
[SUCCESS] Build completed successfully!
[SUCCESS] ==========================================
[INFO] Image file: methyldl_ubuntu22_single.sif
[INFO] Image size: 8.5G
[INFO] Build time: 23m 14s
```

### Test Script Output

```
[INFO] ==========================================
[INFO] MethylDL Container Test Suite
[INFO] ==========================================
[TEST] Check Python version
[✓] Check Python version
[TEST] Import PyTorch
[✓] Import PyTorch
[TEST] Check PyTorch CUDA availability
[✓] Check PyTorch CUDA availability
...
[INFO] ==========================================
[INFO] Test Summary
[INFO] ==========================================
[INFO] Total tests:  25
[✓] Passed:       25
[INFO] Failed:       0
[SUCCESS] All tests passed! ✓
```

## Troubleshooting

### Build Issues

**Problem:** "HPCCM is not installed"
```bash
pip install hpccm
```

**Problem:** "Permission denied" during build
```bash
# Try fakeroot instead of sudo
./build_container.sh recipe_name --fakeroot

# Or use sudo
./build_container.sh recipe_name --sudo
```

**Problem:** Build takes too long
```bash
# Use multi-stage recipes for faster rebuilds (caching)
./build_container.sh methyldl_ubuntu22_multi --fakeroot
```

### Test Issues

**Problem:** GPU tests fail
```bash
# Check NVIDIA drivers
nvidia-smi

# Run container with --nv flag
apptainer exec --nv image.sif nvidia-smi

# Skip GPU tests if not needed
./test_container.sh recipe_name --no-gpu
```

**Problem:** Import tests fail
```bash
# Run verbose to see exact error
./test_container.sh recipe_name --verbose

# Check if packages were installed during build
apptainer exec image.sif poetry show
```

## Advanced Usage

### Custom Directory Structure

```bash
# Override default directories by editing script variables
export RECIPE_DIR="/custom/path/to/recipes"
export DEF_DIR="/custom/path/to/definitions"
export IMAGE_DIR="/custom/path/to/images"

./build_container.sh recipe_name --fakeroot
```

### Parallel Builds

```bash
# Build multiple containers in parallel
./build_container.sh methyldl_ubuntu22_single --fakeroot &
./build_container.sh methyldl_rockylinux9_single --fakeroot &
wait
```

### Automated Testing Pipeline

```bash
#!/bin/bash
# CI/CD pipeline example

RECIPES=("methyldl_ubuntu22_single" "methyldl_rockylinux9_single")

for recipe in "${RECIPES[@]}"; do
    echo "Building ${recipe}..."
    ./build_container.sh "$recipe" --fakeroot --clean
    
    echo "Testing ${recipe}..."
    if ./test_container.sh "$recipe"; then
        echo "✓ ${recipe} passed"
    else
        echo "✗ ${recipe} failed"
        exit 1
    fi
done

echo "All containers built and tested successfully!"
```

## Script Customization

Both scripts are designed to be customizable:

### Modify Build Defaults

Edit `build_container.sh`:
```bash
# Change default directories
RECIPE_DIR="${PROJECT_ROOT}/my_recipes"
DEF_DIR="${PROJECT_ROOT}/my_defs"
IMAGE_DIR="${PROJECT_ROOT}/my_images"

# Change default build method
USE_FAKEROOT=true  # or USE_SUDO=true
```

### Modify Test Defaults

Edit `test_container.sh`:
```bash
# Skip certain test categories by default
TEST_GPU=false
TEST_IMPORTS=false

# Add custom tests
run_test "My custom test" \
    "apptainer exec '$IMAGE_FILE' my-command" \
    "expected-output"
```

## Integration with HPC Systems

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=methyldl_build
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load apptainer

./build_container.sh methyldl_ubuntu22_single --fakeroot
./test_container.sh methyldl_ubuntu22_single
```

### Running on HPC

```bash
# Load container and run with GPU
module load apptainer
apptainer run --nv \
  -B $SCRATCH/data:/workspace/data \
  -B $SCRATCH/outputs:/workspace/outputs \
  container/images/methyldl_ubuntu22_single.sif \
  --config config.yaml
```

## Best Practices

1. **Use version control** for recipe files
2. **Test locally** before deploying to HPC
3. **Use multi-stage builds** for production (smaller images)
4. **Use single-stage builds** for development (faster iteration)
5. **Run full test suite** before deployment
6. **Document custom modifications** in recipes
7. **Keep build logs** for troubleshooting
8. **Use meaningful recipe names** (include OS and version)

## Performance Tips

- **Build time:** Single-stage ~15-20 min, Multi-stage ~25-35 min
- **Image size:** Single-stage ~10-15GB, Multi-stage ~6-10GB
- **Test time:** Full suite ~2-5 min, Quick test ~30 sec
- **Use SSD** for faster builds
- **Allocate 4+ CPU cores** for parallel compilation
- **Ensure 20+ GB free disk space** for build artifacts

## Support

For issues or questions:
1. Run with `--verbose` flag for detailed output
2. Check `TROUBLESHOOTING.md` for common issues
3. Review build logs in terminal output
4. Verify recipe syntax with HPCCM documentation
