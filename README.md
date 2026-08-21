# Syto

![Coverage](coverage-badge.svg) ![Code-style](https://img.shields.io/badge/code%20style-black-black)

Repository for sequence-based long-read combined DNA-methylome classification and deconvolution of cell types using Deep Learning models.

## Installation

Syto is packaged with [Poetry](https://python-poetry.org/): the dependency set,
the `syto` console script and the build backend are all declared in
`pyproject.toml`, and `poetry.lock` pins the exact versions every environment
gets. You need Python 3.11+, Poetry 2.x and [Git LFS](https://git-lfs.com/).

Inference runs on CPU, GPU-trained checkpoints included — models are loaded
onto whichever device the machine has. A CUDA 12.8 GPU is still strongly
recommended: the transformer read classifiers (MethylBERT, EpigenBERT2) are
impractically slow without one and say so in the log when they start on CPU.
Training is GPU-only in practice.

Install Poetry first if you do not have it:

```shell
curl -sSL https://install.python-poetry.org | python3 -
# or, if you use pipx:  pipx install poetry
poetry --version
```

Then clone and install:

```shell
git clone https://github.com/CompEpigen/methyldl.git syto
cd syto

# the bundled foundation models (MethylBERT, DNABERT-2) are stored with Git LFS
git lfs install
git lfs pull

poetry install
poetry run syto --help
```

`poetry install` puts the `syto` command in the project virtualenv. 
Activate the environment once:

```shell
eval "$(poetry env activate)"
syto --help
```

## Quick-start

[`quickstart.sh`](quickstart.sh) downloads one published model run and the data it
needs from the [Syto deposit on Hugging Face](https://huggingface.co/datasets/CompEpigen/syto.1.0),
writes an inference config for one RRBS sample, and deconvolutes it:

```shell
./quickstart.sh
```

It first asks you to log into Hugging Face — a free read token from
<https://huggingface.co/settings/tokens> is enough — then fetches about 5 GB into
`./syto-data`. Override the location with `SYTO_DATA_ROOT=/path/to/data ./quickstart.sh`.

What it downloads, all from the `CompEpigen/syto.1.0` dataset repo:

| Archive | Contents | Size |
| --- | --- | --- |
| `tier1-models/ood-rrbs/…_cancerdetector_uniform_v1.zip` | read classifier, five deconvolvers, their calibrators | 344 MB |
| `tier2-pseudobulks/ood-rrbs/…_cancerdetector_uniform_v1.zip` | pseudobulks the run was fitted on, source of the cell-type prior | 3.2 GB |
| `tier3-training-data/…_atlases_v1.zip` | cell-type atlases (hg19 / hg38) | 4.7 MB |
| `tier4-source-data/rrbs-recovered-reads/…_U250l4hg19_v1.zip` | 522 RRBS samples as recovered reads | 1.2 GB |
| `tier0-results/…_mappings_v1.zip` | `labels_dict.json`, the 39 cell-type label map | 12 KB |

The archives unpack into one tree whose layout the published configs already
expect, so every path in the generated config is relative to `syto-data/` and the
run happens from there. Inference on one sample takes about ten seconds on a GPU and
writes to `syto-data/outputs/quickstart/<sample>/`:

| File | Contents |
| --- | --- |
| `deconvolution_results.csv` | cell-type proportions, one row per cell type × method × calibrator |
| `deconvolution_report.pdf` | the same, as a report; a summary is also printed to the terminal |
| `dmr_aggregated.pkl` | the per-region prediction matrix the deconvolvers consumed |
| `quickstart_inference.yaml` | the config that produced the run |

The quick-start needs no GPU: its `cancer_detector` classifier is a CPU model,
and the neural deconvolvers and calibrators fall back to CPU, so all 20
method x calibrator combinations run either way and give the same numbers.

Rerunning is cheap: the script skips anything already unpacked. To deconvolute a
different sample, point `input.data_path` at another parquet under
`tier4-source-data/rrbs-recovered-reads/U250.l4.hg19/` and rerun from `syto-data`:

```shell
syto inference --config quickstart_inference.yaml
```

To start from your own BAM instead of recovered reads, set `input.type: bam`,
`input.data_path` to the BAM, and add `input.reference_path` (WGBS) or
`input.data_type: ont`. `syto create-config` walks you through a config
interactively.

The hg38 and hg19 references needed to parse BAMs can additionally be downloaded
from the [Syto deposit on Hugging Face](https://huggingface.co/datasets/CompEpigen/syto.1.0).
Each bundle is one ~1.1 GB archive holding the genome FASTA and its index, the
CpG coordinate tracks and the chromosome sizes; unpack it into the same data root
the quick-start uses:

```shell
GENOME=hg38   # or hg19
ARCHIVE="tier4-source-data/reference-genomes/SYTO_tier4_sourcedata_referencegenomes_${GENOME}_v1.zip"

hf download CompEpigen/syto.1.0 "$ARCHIVE" --repo-type dataset --local-dir syto-data/.archives
unzip -qo "syto-data/.archives/$ARCHIVE" -d syto-data
```

Point `input.reference_path` at the unpacked genome, i.e.
`tier4-source-data/reference-genomes/$GENOME/$GENOME.fa.gz` relative to `syto-data`.

## Description

- What is it for.
- Supported models:
    - Classifiers
    - Deconvolvers
    - Calibrators
    - External baselines

## Documentation

- CLI
- Config Wizzard
- Running localy
- Running on HPC

## Citation
