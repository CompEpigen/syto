# Changelog

All notable changes to syto are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-09-15

### Added

- `build_dataset` can stage reads directly from BAM files (`input_type: bam`).
  Each BAM in `input_dir` is labelled with its group from `sample_groups_path`
  (CSV with `name` and `group` columns); BAMs not listed there are skipped.
  Parsing is configured under `bam_processing` and is restricted to the atlas
  regions.
- `atlas_cell_types` option for `build_dataset`, so atlases cell-types can be 
  overwritten at the runtime - useful for bringing custom cohorts for 
  training.
- `UXMMethylationAtlas.from_wgbstools_markers` builds an atlas from the
  `Markers.*.bed` files written by `wgbstools find_markers`.
- Chromosome names are harmonised between BAMs and the atlas, so BAMs aligned to
  references without the `chr` prefix (e.g. hs37d5) work with `chrN` atlases.
- `CHANGELOG.md` documenting releases from 0.5.0 onward.

### Changed

- BAM parsing helpers shared by inference and dataset building now live in
  `syto.data.sequencing.bam_processing`.

### Fixed

- WGBS methylation calls for read 2 of paired-end reads were
  taken from the wrong strand, which made read 2 appear almost fully
  methylated. While this doesn't affect published experiment, where all
  reads were recovered from the .pat with pairs information already
  accounted for, it would have influenced downstream use. Therefore, 
  the users willing to deconvolve their .bam files must use version 1.1.0
  and higher.

### Removed

- `tests/data/test_pseudobulk_conversion.py`, which depended on a script that is
  not part of the repository.

## [1.0.0] - 2026-09-09

First public release under the name syto (previously methyldl).

### Added

- `syto` command-line entry point, replacing the old `App/` scripts.
- Interactive configuration wizard, including a lazy mode, with schemas for
  pseudobulk generation and `fit_deconvolution`.
- `build_dataset` pipeline that turns recovered reads into atlas-overlapped
  Parquet datasets in two phases (`stage`, `finalize`), with `split_remap`, read
  filters, and a separate `staged_source_dir` for finalizing existing data.
- Config-driven labelers for dataset finalization: data-driven soft labels (with
  optional pooling and superset counts), archetype soft labels (
    in development and not published), and hard labels
  with background and smoothing.
- EpiDISH deconvolution baseline (RPC, CBS, CP) integrated into inference, the
  pseudobulk pipeline, and the wizard.
- Abstract base classes for atlases, signatures, distances, and labelers.
- Building a UXM atlas from labelled reads.
- Columnar (Parquet) pseudobulk store and Parquet input at inference, alongside
  HDF5.
- Balanced training (`BalancedTrainer`) and early stopping for MethylBERT and
  EpigenBERT.
- Classifier fitting on columnar data.
- Inference without labels, BAM filtering and chunked processing at inference,
  and non-overlapping chunking of reads longer than the model input length.
- Configurable lower and upper methylation thresholds for ONT data.
- Deconvolution report at inference (opt-in): CLI summary and PDF report.
- Calibration of baseline deconvolvers.
- CPU-only support.
- Classifier calibration metrics and a hyperparameter for
  `ConfidenceWeightedCrossEntropy`.
- Quick-start script, license, and README.

### Changed

- Renamed the package from `methyldl` to `syto`; the `modelling` module is now
  `classification`.
- Renamed `dmr_label_column` to `grg_label_column`.
- Reworked pseudobulk generation and propagated the new pseudobulk format.
- Replaced the classifier adapter with an abstract base class and harmonised the
  classifier API.
- Decoupled baselines from the core pipeline and refactored the experiment
  wrapper.
- Harmonised metric logging for CancerDetector and the lookup classifier.
- MLflow reporting logs train metrics and best-checkpoint metrics, and stores the
  config and log as artifacts.
- Optimized CelFiE EM; new `freeze_gamma` and `sum_by_region` options.

### Fixed

- MethylBERT no longer crashes at inference when the original label column is
  missing.
- Files with inconsistent target names are no longer skipped when a match
  dictionary is provided.
- Shape of the feature matrices when the number of prediction classes differs
  from the number of labels to deconvolve.
- Loading of calibrators and deconvolvers.

### Removed

- `dmr_overlap_analysis` and `split_rebalancing`.
- R code and exploratory data analysis, which moved to a separate repository.
- Obsolete tutorials and local job scripts.

## [0.5.0] - 2026-05-18

First tagged release (as methyldl).

### Added

- Read-level classifiers: core model, DISMIR, MethylBERT, mini-RNNs, and a lookup
  classifier with hard labels.
- CancerDetector baseline.
- Soft labels for training.
- Deconvolvers: NNLS, SVM, MLE, MLP, and SWN, with a harmonised API.
- Calibrators (including Dirichlet calibration) in a separate module.
- Independent cross-validation engine and a combined deconvolution loss.
- Bootstrap confidence intervals.
- End-to-end inference integration.

[Unreleased]: https://github.com/CompEpigen/syto/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/CompEpigen/syto/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/CompEpigen/syto/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/CompEpigen/syto/releases/tag/v0.5.0
