# Syto CLI

## Commands

Every command runs one task from one YAML config, except `syto create-config`:

```bash
syto <command> --config <your-config.yaml>
```

| Command | Purpose |
| --- | --- |
| `syto create-config` | generates a config interactively (usefull for the new users) |
| `syto build-dataset` | builds labelled Parquet dataset from recovered .pat reads|
| `syto classifier-fit` | fits a read-level methylation classifier |
| `syto generate-pseudobulk` | runs classfication on the labeled Parquet dataset and generates pseudobulks from obtained predictions|
| `syto fit-deconvolution` | fit deconvolvers (XGB, SWN, MLP, NNLS, PSLS) on pseudobulks |
| `syto deconvolute-pseudobulk` | runs external baselines over a target pseudobulks to generate corresponding deconvolution results |
| `syto fit-calibration` | fits calibrators on deconvolver predictions (can be used for syto and supported external baselines alike) |
| `syto confidence-intervals` | recomputes pseudobulk deconvolution metrics with bootstrap confidence intervals |
| `syto inference` | deconvolutes a single sample (BAM or pre-classified reads) |

`syto pretrain` is reserved for the future (we might add functionality to pretrain foundational models behind selected classifiers).
At the present moment, it raises `NotImplementedError`. 

Run `syto --help` for the list, `syto <command> --help` for its flags.

## Configs

The eathiest way to start with configs is to use config wizzard. 

```bash
syto create-config
```

It offers two modes: *templates*, which writes one skeleton per task with
`<angle-bracketed>` placeholders to fill in later, and *fine-grained*, which
allows you to fill it in step by step fashion from the command line. 
The *fine-grained* also validates the paths to classifiers,deconvolvers,etc.
*templates* mode is more convinient and fast for seasonal users, but 
if you are new to syto and want to learn about configuration parameters, 
*fine-grained* might be a better choice, albeit much slower.

To check a hand-written config without starting a run:

```bash
syto fit-calibration --config my_calibration.yaml --dry-run
```

### Required keys

Validation happens before any pipeline is constructed, so a bad config will fail to execute.


| Command | Required config keys |
| --- | --- |
| `build-dataset` | `input_dir`, `output_dir`, `atlas_path`, `reference_genome`, `labels_dict_path`, `n_buckets`, `signature`, `labelers` |
| `classifier-fit` | `model`, `data_path`, `max_sequence_length` |
| `generate-pseudobulk` | `output_dir`, `labels_dict_path` |
| `fit-deconvolution` | `pseudobulk_path`, `output_dir`, `labels_dict_path` |
| `deconvolute-pseudobulk` | `pseudobulk_path`, `output_dir`, `labels_dict_path` |
| `fit-calibration` | `output_dir`, `labels_dict_path` |
| `confidence-intervals` | `calibration_results_dir`, `labels_dict_path` |
| `inference` | `labels_dict_path`, `input` |

Two things the table does not show:

- **`output_dir` is required by every command**, including the ones above that
  do not list it. The CLI copies your config into it before running, so the run
  is reproducible from its own output directory.
- **`fit-calibration` additionally requires `pseudobulk_path`,
  `features_mask_path`, and `deconvolvers_dir`** whenever a deconvolver has no
  `predictions_dir` — that is, whenever a saved model has to be loaded and
  evaluated.

`labels_dict_path` may point at the mapping bundled with the package,
`syto/app/labels_dict.json`, or at your own.

The rest of the parameters are applied from defaults if no explicit configuration is provided.
You are highly encoraged to fill in all config parameters, overwriting the defaults if nessesary, but 
for convinience the required list is kept as small as possible. 

## Flags

Available on every command:

| Flag | Effect |
| --- | --- |
| `--config PATH` | The YAML config to run (required) |
| `--dry-run` | Validate the config and exit without running |
| `--verbose` | Debug-level logging |
| `--log-file PATH` | Also write logs to a file (default: stdout only) |

Overrides, applied on top of the config before validation:

| Flag | Config key it sets |
| --- | --- |
| `--model {dismir,methylbert,cancer_detector,lookup,epigenbert2}` | `model.architecture` |
| `--data-path PATH` | `data_path` |
| `--max-seq-length N` | `max_sequence_length` |
| `--checkpoint PATH` | `checkpoint_path` |
| `--bam PATH` | `input.type: bam` + `input.bam_path` |
| `--atlas PATH` | `atlas_path` |
| `--labels-dict PATH` | `labels_dict_path` |
| `--output-dir PATH` | `output_dir` |
| `--datasets A B C` | `datasets` |
| `--mlflow-uri URI` | `mlflow.tracking_uri` |
| `--experiment-name NAME` | `mlflow.experiment_name` |

Overrides are how you sweep one config across many samples without writing a
config per sample, which is usefull in scenarious like inference. 

```bash
for bam in samples/*.bam; do
    syto inference --config inference.yaml \
        --bam "$bam" \
        --output-dir "results/$(basename "$bam" .bam)"
done
```

### Deprecated `--task`

You can also run specific tool with a `--task` predicate, but this option will be 
removed in the future. 

```bash
syto --task fit_calibration --config c.yaml   # → syto fit-calibration --config c.yaml
```

## Layout
```
syto/app/
├── cli.py                                # `syto` entry point (console script)
├── dataset_build_pipeline.py             # syto build-dataset
├── classifier_fit_pipeline.py            # syto classifier-fit
├── pseudobulk_pipeline.py                # syto generate-pseudobulk
├── deconvolution_pipeline.py             # syto fit-deconvolution
├── pseudobulk_deconvolution_pipeline.py  # syto deconvolute-pseudobulk
├── calibration_pipeline.py               # syto fit-calibration
├── conf_interval_pipeline.py             # syto confidence-intervals
├── inference.py                          # syto inference
├── labels_dict.json                      # default label mapping used in the paper
├── wizard/                               # syto create-config
└── Readme.md                             # This file
```
