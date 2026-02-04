# MethylDL HPCCM Recipes - Troubleshooting Guide

## Common Build Issues and Solutions

### Issue 1: GPG Agent Error (Ubuntu)

**Error:**
```
gpg: error running '/usr/bin/gpg-agent': probably not installed
add-apt-repository failed with exit status 1
```

**Cause:** The `gpg-agent` package is not installed, which is required by `add-apt-repository` to verify PPA signatures.

**Solution:** ✅ **FIXED** - The recipes now install `gnupg` and `gpg-agent` before adding the deadsnakes PPA.

### Issue 2: Python 3.12 Not Available

**Error:**
```
E: Unable to locate package python3.12
```

**Cause:** The deadsnakes PPA was not added successfully or apt cache needs updating.

**Solution:** The recipes include proper sequencing:
```python
# Install gnupg first
Stage0 += apt_get(ospackages=['software-properties-common', 'gnupg', 'gpg-agent', ...])

# Then add PPA
Stage0 += shell(commands=['add-apt-repository ppa:deadsnakes/ppa -y', 'apt-get update', ...])
```

### Issue 3: Poetry Installation Fails

**Error:**
```
curl: (6) Could not resolve host: install.python-poetry.org
```

**Cause:** Network access is blocked or DNS resolution fails during build.

**Solution:** Check your network settings. For Apptainer/Singularity:
```bash
# Build with network access
apptainer build --network methyldl.sif methyldl_ubuntu22.def

# Or use fakeroot if needed
apptainer build --fakeroot --network methyldl.sif methyldl_ubuntu22.def
```

### Issue 4: CUDA Libraries Not Found

**Error:**
```
ImportError: libcudart.so.12: cannot open shared object file
```

**Cause:** Running container without GPU support.

**Solution:**
```bash
# Singularity/Apptainer - use --nv flag
apptainer run --nv methyldl.sif

# Docker - use --gpus all
docker run --gpus all methyldl:ubuntu22
```

### Issue 5: Permission Denied During Build

**Error:**
```
FATAL: Unable to create build: permission denied
```

**Solution:**
```bash
# Option 1: Use fakeroot (no sudo)
apptainer build --fakeroot methyldl.sif methyldl_ubuntu22.def

# Option 2: Use sudo
sudo apptainer build methyldl.sif methyldl_ubuntu22.def
```

### Issue 6: Poetry Lock File Missing

**Error:**
```
Error: poetry.lock not found
```

**Solution:**
```bash
# Generate lock file before building
poetry lock

# Or modify recipe to skip lock file
# Remove this line from recipe:
# Stage0 += copy(src='./poetry.lock', dest='/workspace/methyldl/poetry.lock')
```

### Issue 7: Rocky Linux - Python Build Fails

**Error:**
```
configure: error: C compiler cannot create executables
```

**Cause:** Missing development tools or libraries.

**Solution:** The recipes include all required build dependencies:
```python
Stage0 += shell(commands=[
    'dnf -y install gcc gcc-c++ make',
    'dnf -y install openssl-devel libffi-devel zlib-devel bzip2-devel'
])
```

### Issue 8: Multi-Stage Build - File Not Found

**Error:**
```
COPY failed: stat /build/methyldl/.venv: no such file or directory
```

**Cause:** Build stage didn't complete successfully.

**Solution:** Use single-stage build for debugging first:
```bash
# Try single-stage first
hpccm --recipe methyldl_ubuntu22_single.py --format singularity > test.def
apptainer build test.sif test.def

# Once working, use multi-stage for production
hpccm --recipe methyldl_ubuntu22_multi.py --format singularity > prod.def
```

## Build Best Practices

### 1. Generate Definition First
```bash
# Generate and inspect before building
hpccm --recipe methyldl_ubuntu22_single.py --format singularity > methyldl.def
less methyldl.def  # Review the definition
apptainer build methyldl.sif methyldl.def
```

### 2. Use Single-Stage for Development
```bash
# Faster iterations, easier debugging
hpccm --recipe methyldl_ubuntu22_single.py --format singularity > dev.def
apptainer build --fakeroot dev.sif dev.def
```

### 3. Use Multi-Stage for Production
```bash
# Smaller images, optimized
hpccm --recipe methyldl_ubuntu22_multi.py --format singularity > prod.def
sudo apptainer build prod.sif prod.def
```

### 4. Test Locally Before HPC Deployment
```bash
# Test on workstation first
apptainer shell --nv methyldl.sif
Apptainer> python3 --version
Apptainer> python3 -c "import torch; print(torch.cuda.is_available())"
```

## Verification Commands

### Check Python Installation
```bash
apptainer exec methyldl.sif python3 --version
# Should output: Python 3.12.x
```

### Check PyTorch and CUDA
```bash
apptainer exec --nv methyldl.sif python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Check Poetry Environment
```bash
apptainer exec methyldl.sif poetry --version
apptainer exec methyldl.sif poetry env info
```

### List Installed Packages
```bash
apptainer exec methyldl.sif poetry show
```

## Recipe Customization

### Disable nvtop (if causing issues)
```python
# Remove nvtop from package list
Stage0 += apt_get(ospackages=[
    'vim',
    'htop',
    # 'nvtop',  # Comment out if problematic
    ...
])
```

### Change Python Version
```python
# For Python 3.11 instead of 3.12
Stage0 += shell(commands=[
    'add-apt-repository ppa:deadsnakes/ppa -y',
    'apt-get update',
    'apt-get install -y python3.11 python3.11-venv python3.11-dev',
    'update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1',
    ...
])
```

### Add Additional Packages
```python
# Add to apt_get ospackages list
Stage0 += apt_get(ospackages=[
    'vim',
    'htop',
    'tmux',        # Add terminal multiplexer
    'screen',      # Add screen
    'ncdu',        # Add disk usage analyzer
    ...
])
```

## Getting Help

If issues persist:

1. **Check HPCCM version:**
   ```bash
   hpccm --version
   ```

2. **Validate recipe syntax:**
   ```bash
   python3 methyldl_ubuntu22_single.py
   ```

3. **Generate definition and inspect:**
   ```bash
   hpccm --recipe methyldl_ubuntu22_single.py --format singularity > test.def
   cat test.def
   ```

4. **Build with verbose output:**
   ```bash
   apptainer build --debug methyldl.sif methyldl.def 2>&1 | tee build.log
   ```

5. **Check base image availability:**
   ```bash
   docker pull nvcr.io/nvidia/nvhpc:25.3-runtime-cuda12.8-ubuntu22.04
   ```

## Quick Reference

| Issue | Ubuntu Fix | Rocky Linux Fix |
|-------|------------|-----------------|
| GPG error | Install `gnupg` + `gpg-agent` | N/A (uses dnf) |
| Python 3.12 | Use deadsnakes PPA | Build from source |
| Missing packages | `apt-get install` | `dnf install` |
| Poetry fails | Check network | Check network |
| CUDA not found | Use `--nv` flag | Use `--nv` flag |

## Updated Recipes (2025-10-22)

✅ **Fixed:** Ubuntu recipes now include `gnupg` and `gpg-agent`
✅ **Fixed:** Proper package installation order
✅ **Tested:** Base images from NVIDIA NGC catalog
✅ **Verified:** Python 3.12 installation from deadsnakes PPA
