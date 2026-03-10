#!/usr/bin/env python
"""Recipe for MethylDL Singularity container - Multi-stage Ubuntu 22.04
Optimized for smaller image size with separate build and runtime stages

Usage:
    $ hpccm --recipe methyldl_ubuntu22.04_single.py --format docker >> methyldl_ubuntu22.04_single.docker
    $ hpccm --recipe methyldl_ubuntu22.04_single.py --format singularity >> methyldl_ubuntu22.04_single.def
"""

# ============= Stage 0: Builder =============
# Base image with CUDA 12.8 support
Stage0 += baseimage(image="nvcr.io/nvidia/nvhpc:25.3-runtime-cuda12.8-ubuntu22.04")

# Update and install system dependencies
Stage0 += apt_get(
    ospackages=[
        "software-properties-common",
        "gnupg",
        "gpg-agent",
        "curl",
        "git",
        "build-essential",
        "wget",
        "vim",
        "htop",
        "nvtop",
        "libgomp1",
        "libopenmpi-dev",
        "openmpi-bin",
    ]
)

# Add Python 3.12 from deadsnakes PPA
Stage0 += shell(
    commands=[
        "add-apt-repository ppa:deadsnakes/ppa -y",
        "apt-get update",
        "apt-get install -y python3.12 python3.12-venv python3.12-dev",
        "update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1",
        "update-alternatives --set python3 /usr/bin/python3.12",
        "curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12",
    ]
)

# Install Poetry
Stage0 += shell(
    commands=[
        "curl -sSL https://install.python-poetry.org | python3.12 -",
        "ln -s /root/.local/bin/poetry /usr/local/bin/poetry",
        "poetry config virtualenvs.in-project true",
        "poetry config installer.max-workers 10",
    ]
)

# Create workspace directory
Stage0 += shell(commands=["mkdir -p /workspace/methyldl"])

# Create test_container_tmp directory
Stage0 += shell(commands=["mkdir -p /workspace/methyldl/test_container_tmp"])

# Copy package files (dependency files first for better caching)
Stage0 += copy(src="./pyproject.toml", dest="/workspace/methyldl/pyproject.toml")
Stage0 += copy(src="./poetry.lock", dest="/workspace/methyldl/poetry.lock")

# Set working directory
Stage0 += workdir(directory="/workspace/methyldl")

# Install dependencies first (better caching)
Stage0 += shell(
    commands=[
        "cd /workspace/methyldl",
        "poetry install --no-root --no-interaction --no-ansi",
        "poetry cache clear pypi --all -n",
    ]
)

# Copy application code
Stage0 += copy(src="./methyldl", dest="/workspace/methyldl/methyldl")
Stage0 += copy(src="./App", dest="/workspace/methyldl/App")
Stage0 += copy(src="./README.md", dest="/workspace/methyldl/README.md")
Stage0 += copy(
    src="./foundationalModels", dest="/workspace/methyldl/foundationalModels"
)
Stage0 += copy(src="./tests", dest="/workspace/methyldl/tests")
Stage0 += copy(src="./EDA/edautils.py", dest="/workspace/EDA/edautils.py")

# Install the package itself
Stage0 += shell(
    commands=[
        "cd /workspace/methyldl",
        "poetry install --only-root --no-interaction --no-ansi",
    ]
)

# Set environment variables
Stage0 += environment(
    variables={
        "PATH": "/workspace/methyldl/.venv/bin:$PATH",
        "VIRTUAL_ENV": "/workspace/methyldl/.venv",
        "PYTHONPATH": "/workspace/methyldl:$PYTHONPATH",
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST": '"7.5;8.0;8.6;8.9;9.0"',
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "TRANSFORMERS_CACHE": "/workspace/cache/huggingface",
        "HF_HOME": "/workspace/cache/huggingface",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }
)

# Create cache and data directories
Stage0 += shell(
    commands=[
        "mkdir -p /workspace/cache/huggingface",
        "mkdir -p /workspace/data",
        "mkdir -p /workspace/outputs",
    ]
)

# Add metadata labels
Stage0 += label(
    metadata={
        "maintainer": "Dmytro Rizdvanteskyi",
        "description": "MethylDL container - Ubuntu 22.04",
        "cuda.version": "12.8",
        "pytorch.version": "2.7.0+cu128",
        "python.version": "3.12",
        "os": "ubuntu22.04",
    }
)

# Define runscript
Stage0 += runscript(
    commands=[
        "#!/bin/bash",
        'echo "MethylDL Container - PyTorch $(python3 -c \\"import torch; print(torch.__version__)\\" 2>/dev/null || echo \\"loading...\\")"',
        'echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo \\"N/A\\")"',
        'echo "Python: $(python3 --version)"',
        "cd /workspace/methyldl",
        'poetry run python App/main.py "$@"',
    ]
)

# Add help section for Singularity
Stage0 += shell(
    commands=[
        "mkdir -p /.singularity.d",
        'echo "#!/bin/bash" > /.singularity.d/runscript.help',
        'echo "MethylDL Singularity Container (Ubuntu 22.04)" >> /.singularity.d/runscript.help',
        'echo "=============================================" >> /.singularity.d/runscript.help',
        'echo "" >> /.singularity.d/runscript.help',
        'echo "Usage:" >> /.singularity.d/runscript.help',
        'echo "  singularity run --nv methyldl.sif [arguments]" >> /.singularity.d/runscript.help',
        'echo "" >> /.singularity.d/runscript.help',
        'echo "Bind paths for data:" >> /.singularity.d/runscript.help',
        'echo "  -B /path/to/data:/workspace/data" >> /.singularity.d/runscript.help',
        'echo "  -B /path/to/outputs:/workspace/outputs" >> /.singularity.d/runscript.help',
        'echo "  -B /path/to/cache:/workspace/cache" >> /.singularity.d/runscript.help',
    ]
)
