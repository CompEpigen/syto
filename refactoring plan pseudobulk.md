# Plan: Pseudobulk Generator Refactoring

## TL;DR
Refactor `pseudo_bulk_generation.py` into a cleaner `PseudobulkGenerator` class that outputs a self-contained HDF5 file with full reproducibility metadata, inputs, parameters, and outputs (pseudobulks + pure profiles per split). Use h5py for HDF5 writing and Dask for parallelization.

---

## Steps

### Phase 1: Data Model & HDF5 Schema Setup

1. **Define HDF5 schema constants** — Create a module with HDF5 group/dataset path constants matching the diagram structure (metadata, inputs, parameters, outputs). *No dependencies*

2. **Create HDF5Writer utility class** — Wrapper around h5py for creating groups, writing datasets with compression, and handling metadata attributes. *Depends on 1*

3. **Define dataclasses for intermediate results** — `PseudobulkResult`, `PureProfileResult`, `GenerationMetadata` to structure outputs before HDF5 serialization. *No dependencies, parallel with 1-2*

### Phase 2: Core Generation Logic Refactoring

4. **Refactor `generate_single_pseudobulk` to accept seed parameter** — Make the seed explicit (not randomly generated inside) for reproducibility. Store seed + sampling function reference in output. *Depends on 3*

5. **Add substitution method support** — Integrate `_fill_in_missing_labels()` and `_apply_prior_substitution()` from `prediction_aggregation.py` into the feature matrix generation. Support uniform_number, prior_blending, prior_imputation. *Depends on 4*

6. **Implement `_generate_single_split` method** — Complete the placeholder. Should iterate over target proportions, call `generate_single_pseudobulk`, collect results for a single split. *Depends on 4, 5*

### Phase 3: Dask Parallelization & Checkpointing

7. **Create Dask-based parallel generation** — Replace ProcessPoolExecutor pattern with Dask Delayed for lazy pseudobulk generation. Each pseudobulk = one delayed task. *Depends on 6*

8. **Implement crash-resilient batch file approach** — Instead of appending to single HDF5:
   - Write each batch to **separate small HDF5 files**: `{output_dir}/batches/batch_{idx:06d}.h5`
   - Each batch file is written atomically (temp file + rename)
   - Checkpoint file (`{output}.checkpoint.json`) tracks which batches are complete
   - If crash during batch write: only that one batch file may be corrupted, all previous batches are safe
   *Depends on 2*

9. **Implement batch processing with atomic writes** — Use Dask's compute with batching. For each batch:
   1. Generate pseudobulks in memory
   2. Write to temp file `batch_{idx}.h5.tmp`
   3. Rename to `batch_{idx}.h5` (atomic)
   4. Update checkpoint file atomically
   *Depends on 7, 8*

10. **Implement final consolidation step** — After all batches complete:
   1. Read all batch files
   2. Write consolidated final HDF5 file (as single operation)
   3. If consolidation succeeds: delete batch files + checkpoint
   4. If crash during consolidation: batch files intact, can retry
   *Depends on 9*

### Phase 4: Pure Profile Integration

12. **Integrate pure profile generation** — Add method `_generate_pure_profiles_for_split()` that creates 39 pure profiles (one per cell type) per split. Store in batch files alongside pseudobulks. *Depends on 6*

13. **Compute and store uniform prior matrix** — After pure profiles, compute the averaged uniform prior matrix per split. *Depends on 12*

### Phase 5: HDF5 Consolidation & Output Assembly

14. **Implement `_consolidate_batches`** — Read all batch files, merge into final HDF5:
   - Verify all batches are present and valid (checksum/structure validation)
   - Write `/inputs/` group (metadata + full DataFrames)
   - Write `/parameters/` group
   - Write `/outputs/` group (all pseudobulks + pure profiles)
   - Single write operation minimizes corruption window
   *Depends on 2, 10, 13*

15. **Implement `_cleanup_batch_files`** — After successful consolidation:
   - Delete `batches/` directory
   - Delete checkpoint file
   - Log final HDF5 path and stats
   *Depends on 14*

### Phase 6: Main Entry Point

16. **Implement `_check_and_resume`** — Check if checkpoint file exists. If yes:
   - Read completed batch indices
   - Validate batch files exist and are not corrupted
   - Return resume info (start from first missing/corrupted batch)
   *Depends on 8*

17. **Complete `run()` method** — Orchestrate:
   1. Load/create overall config
   2. For each split in order (train → valid → test):
      a. Check split checkpoint for resume
      b. Generate batches (parallel within split, resumable)
      c. Generate pure profiles for split
      d. Mark split as completed
   3. Consolidate all splits into final HDF5
   4. Cleanup all batch directories + checkpoints
   *Depends on all above*

---

## Relevant Files

- [syto/data/pseudobulk_generator.py](syto/data/pseudobulk_generator.py) — Main file to complete; placeholders at `run()`, `_generate_single_split()`, `_aggregate_hdf5_and_cleanup()`
- [syto/data/pseudo_bulk_generation.py](syto/data/pseudo_bulk_generation.py) — Legacy code to reference for `generate_pseudo_bulk_optimized()`, `_compute_samples_per_dmr_multinomial()`, worker patterns
- [syto/modelling/prediction_aggregation.py](syto/modelling/prediction_aggregation.py) — `aggregate_predictions_by_grg_optimized()`, `_fill_in_missing_labels()`, `_apply_prior_substitution()` for substitution strategies
- [syto/data/pure_profile_generation.py](syto/data/pure_profile_generation.py) — `generate_pure_profiles()`, `extract_pure_feature_matrix()`, `compute_uniform_prior_matrix()` to reuse

---

## File Structure During Generation

```
{output_dir}/
├── train/
│   ├── batches/
│   │   ├── batch_000000.h5         # Pseudobulks 0-99 for train split
│   │   ├── batch_000001.h5
│   │   └── ...
│   └── checkpoint.json             # Train-specific progress
├── valid/
│   ├── batches/
│   │   └── ...
│   └── checkpoint.json             # Valid-specific progress
├── test/
│   ├── batches/
│   │   └── ...
│   └── checkpoint.json             # Test-specific progress
├── generation_config.json          # Overall config + parameters hash
└── pseudobulk.h5                   # Final consolidated output

# Per-split checkpoint file (e.g., train/checkpoint.json):
{
  "split": "train",
  "status": "in_progress",  # in_progress | completed
  "total_batches": 300,
  "batch_size": 100,
  "completed_batches": [0, 1, 2, 3, ...],
  "pure_profiles_done": false
}

# Overall config (generation_config.json):
{
  "generation_started_at": "2026-05-26T10:30:00Z",
  "parameters_hash": "sha256:...",
  "splits_order": ["train", "valid", "test"],
  "splits_completed": ["train"],  # Tracks which splits are fully done
  "n_pseudobulks_per_split": {
    "train": 30000,
    "valid": 5000,
    "test": 5000
  }
}
```

**Workflow**: Generate train (all batches) → Generate valid (all batches) → Generate test (all batches) → Consolidate all into final HDF5

## Final HDF5 Structure (after consolidation)

```
/pseudobulk.h5
├── inputs/
│   ├── metadata/
│   │   ├── gr_id_column (attr)
│   │   ├── labeling_scheme (attr)
│   │   ├── classifier (attr)
│   │   ├── data_stats (dataset, optional)
│   │   └── data_watermark (attr)
│   ├── train/ (dataset: read_id, predictions, ncpgs)
│   ├── valid/
│   └── test/
├── parameters/
│   ├── cell_types_mapping (dataset: key-value with ontology)
│   ├── gr_groups (dataset: gr_id, gr_name)
│   ├── substitution_method (attr: uniform_number|prior_blending|prior_imputation)
│   └── gr_sampling_method (attr: uniform_multinomial)
└── outputs/
    ├── train/
    │   ├── pseudobulks/
    │   │   ├── i_0/
    │   │   │   ├── target_proportions (dataset)
    │   │   │   ├── actual_proportions (dataset)
    │   │   │   ├── sampled_indexes/
    │   │   │   │   ├── seed (attr)
    │   │   │   │   └── sampling_function (attr: code reference)
    │   │   │   ├── n_reads_per_gr (dataset)
    │   │   │   └── aggregated_features (dataset)
    │   │   └── ...
    │   └── pure_profiles/
    │       ├── feature_matrices (dataset: shape 39x39)
    │       └── uniform_prior (dataset)
    ├── valid/ (same structure)
    └── test/ (same structure)
```

---

## Verification

1. **Unit tests for HDF5Writer** — Test group creation, dataset writing with compression, attribute setting
2. **Integration test for single pseudobulk generation** — Verify `generate_single_pseudobulk` returns correct structure with seed-based reproducibility
3. **Test substitution methods** — Verify uniform_number, prior_blending, prior_imputation produce expected outputs
4. **Test HDF5 structure compliance** — Read back generated HDF5, validate all groups/datasets exist with correct shapes
5. **Test Dask parallel execution** — Generate small batch (10 pseudobulks), verify no race conditions, correct aggregation
6. **Test pure profile integration** — Verify 39 pure profiles generated per split, uniform prior computed correctly
7. **Test checkpoint/resume** — Abort generation mid-way (e.g., after 50% of pseudobulks), restart with same parameters, verify it resumes from last checkpoint and final output is consistent
8. **Test resume with parameter mismatch detection** — Attempt resume with different parameters, verify it raises an error or warns
9. **End-to-end test with legacy comparison** — Generate pseudobulks with both old and new code, compare numeric outputs for consistency

---

## Decisions

- **h5py** selected over pandas HDFStore for fine-grained control over nested HDF5 groups
- **Dask** selected for parallelization (lazy evaluation, better scaling, HDF5 integration)
- **Full input data storage** in HDF5 for complete reproducibility
- **All three substitution methods** supported: uniform_number, prior_blending, prior_imputation
- **Pure profiles stored per-split** under `outputs/{split}/pure_profiles/`
- **Metadata moved under inputs/** as it describes input data properties
- **Sequential split generation**: train → valid → test, each with its own target proportions and checkpoint folder
- **Per-split checkpointing**: each split has its own `batches/` folder and `checkpoint.json`, enabling independent resume
- **Batch file approach** for crash resilience: write intermediate batch HDF5 files (atomic via temp+rename), consolidate at end

---

## Further Considerations

1. **Compression strategy**: Use gzip compression level 4 (balance speed/size) for large datasets (aggregated_features, inputs). Should specific datasets use different compression? *Recommend: gzip level 4 for all*

2. **Chunking for incremental writes**: HDF5 datasets will be chunked to allow appending pseudobulks incrementally. What chunk size? *Recommend: 100 pseudobulks per chunk*

3. **Legacy compatibility**: Should the new generator also support outputting the old .pkl/.npz format for backward compatibility? *Recommend: No, clean break to HDF5 only*
