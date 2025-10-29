# Quick Reference Card

## Project Structure

```
project/
├── container/
│   ├── hpccm/                    # Recipe files (*.py)
│   ├── definitions/              # Generated definitions (*.def)
│   ├── images/                   # Built images (*.sif)
│   └── scripts/                  # ← Scripts live here!
│       ├── build_container.sh
│       └── test_container.sh
├── methyldl/                     # Python package
├── App/                          # Application
└── pyproject.toml
```

## Path Logic

The scripts automatically detect their location:
- Script location: `container/scripts/`
- Sets `CONTAINER_DIR` = `container/`
- Recipe dir: `container/hpccm/`
- Def dir: `container/definitions/`
- Image dir: `container/images/`

## Quick Commands

```bash
# Navigate to scripts
cd container/scripts

# Build
./build_container.sh methyldl_ubuntu22_single --fakeroot

# Test
./test_container.sh methyldl_ubuntu22_single

# Run (from project root)
cd ../..
apptainer run --nv container/images/methyldl_ubuntu22_single.sif
```

## Available Recipes

- `methyldl_ubuntu22_single` - Ubuntu 22.04, single-stage
- `methyldl_ubuntu22_multi` - Ubuntu 22.04, multi-stage (optimized)
- `methyldl_rockylinux9_single` - Rocky Linux 9, single-stage
- `methyldl_rockylinux9_multi` - Rocky Linux 9, multi-stage (optimized)

## Common Workflows

### First Time Setup
```bash
cd container/scripts
chmod +x *.sh
./build_container.sh methyldl_ubuntu22_single --fakeroot
./test_container.sh methyldl_ubuntu22_single
```

### After Recipe Changes
```bash
cd container/scripts
./build_container.sh methyldl_ubuntu22_single --fakeroot --clean
./test_container.sh methyldl_ubuntu22_single --quick
```

### Production Build
```bash
cd container/scripts
./build_container.sh methyldl_ubuntu22_multi --sudo --clean
./test_container.sh methyldl_ubuntu22_multi
```

## Script Options

### build_container.sh
| Option | Description |
|--------|-------------|
| `--fakeroot` | Build without sudo (default) |
| `--sudo` | Build with sudo |
| `--clean` | Remove old files first |
| `--skip-generate` | Use existing .def |
| `--verbose` | Show details |

### test_container.sh
| Option | Description |
|--------|-------------|
| `--quick` | Fast test only |
| `--no-gpu` | Skip GPU tests |
| `--no-poetry` | Skip Poetry tests |
| `--verbose` | Show details |

## Troubleshooting

### Scripts can't find recipes
```bash
# Make sure you're in the right directory
cd container/scripts
pwd  # Should show: /path/to/project/container/scripts
```

### Build fails
```bash
# Try verbose mode
./build_container.sh recipe_name --fakeroot --verbose

# Or clean build
./build_container.sh recipe_name --fakeroot --clean
```

### Tests fail
```bash
# Run verbose to see details
./test_container.sh recipe_name --verbose

# Quick test only
./test_container.sh recipe_name --quick
```

## File Locations After Build

| File Type | Location | Example |
|-----------|----------|---------|
| Recipe | `container/hpccm/` | `methyldl_ubuntu22_single.py` |
| Definition | `container/definitions/` | `methyldl_ubuntu22_single.def` |
| Image | `container/images/` | `methyldl_ubuntu22_single.sif` |

## Running Containers

### From Project Root
```bash
apptainer run --nv container/images/methyldl_ubuntu22_single.sif
```

### From Anywhere
```bash
apptainer run --nv /full/path/to/project/container/images/methyldl_ubuntu22_single.sif
```

### With Bind Mounts
```bash
apptainer run --nv \
  -B ./data:/workspace/data \
  -B ./outputs:/workspace/outputs \
  container/images/methyldl_ubuntu22_single.sif
```

## Time Estimates

- **Build time:** 15-30 minutes
- **Test time:** 2-5 minutes (full), 30 seconds (quick)
- **Image size:** 6-15 GB depending on recipe

## Support Files

- `README_HPCCM.md` - Recipe documentation
- `SCRIPT_USAGE.md` - Detailed script guide
- `TROUBLESHOOTING.md` - Common issues
