# Promote Balanced/Aux-Loss HF Training to EpigenBERT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move MethylBERT's HF-Trainer-based class-balancing and auxiliary-CE-loss-logging machinery into a shared module and wire it into `EpigenDnabert2`, so both HuggingFace-Trainer classifiers can mitigate class imbalance and trace cross-entropy.

**Architecture:** Extract the three already-architecture-neutral classes (`BalancedBackgroundBatchSampler`, `MethylBertTrainer`, `BalancedTrainer`) from `syto/classification/classifiers/methylbert.py` into a new shared module `syto/classification/hf_training.py`, rename `MethylBertTrainer` to `AuxLossLoggingTrainer` (keeping a back-compat alias), re-export the old names from `methylbert.py` so nothing downstream breaks, then thread `signal_mask`/`bg_ratio` through `EpigenDnabert2._init_trainer` → `fine_tune` → `fit_classificaton` exactly as `MethylBert` already does. Dismir is explicitly out of scope: it uses a hand-written PyTorch loop with a chunk-aware sampler, not an HF Trainer, so these classes do not transfer.

**Tech Stack:** Python, PyTorch, HuggingFace `transformers` (`Trainer`, `Sampler`), `unittest` (run via `python -m unittest`), numpy.

## Global Constraints

- Test runner: `python -m unittest <dotted.path>` (project configures `-m unittest discover -v`). Do **not** assume pytest fixtures.
- Preserve all existing public import paths: `from syto.classification.classifiers.methylbert import MethylBertTrainer, BalancedTrainer, BalancedBackgroundBatchSampler, extract_signal_mask` must keep working.
- `extract_signal_mask` stays in `methylbert.py` — it reads MethylBERT-dataset internals (`lazy_tokenization`, `raw_lines`, `headers`, `lines`) and is NOT generalized here.
- The reusable classes already accept a plain boolean `signal_mask` numpy array indexed by dataset position; keep that contract. Callers build the mask.
- No behavior change for the existing MethylBERT path — this is a move + re-export, verified by the existing MethylBERT suite staying green.
- No git co-author trailer on commits (per project convention).

---

### Task 1: Extract shared HF training module with characterization tests

**Files:**
- Create: `syto/classification/hf_training.py`
- Create: `tests/classification/test_hf_training.py`
- Modify: `syto/classification/classifiers/methylbert.py` (remove three class defs at lines 195-337; add import + re-export)

**Interfaces:**
- Produces (importable from `syto.classification.hf_training`):
  - `class BalancedBackgroundBatchSampler(Sampler)` — `__init__(self, signal_mask, batch_size, bg_ratio=0.3, shuffle=True, drop_last=False)`, `__iter__` yields `list[int]`, `__len__() -> int`.
  - `class AuxLossLoggingTrainer(transformers.Trainer)` — logs `loss_ce` from `outputs.loss_ce` when present (guarded by `hasattr`).
  - `class BalancedTrainer(AuxLossLoggingTrainer)` — `__init__(self, *args, signal_mask=None, bg_ratio=0.3, **kwargs)`, overrides `get_train_dataloader`.
  - `MethylBertTrainer = AuxLossLoggingTrainer` (back-compat alias).
- Produces (still re-exported from `syto.classification.classifiers.methylbert`): `BalancedBackgroundBatchSampler`, `MethylBertTrainer`, `BalancedTrainer`.

- [ ] **Step 1: Write failing characterization tests for the sampler**

Create `tests/classification/test_hf_training.py`:

```python
import unittest

import numpy as np

from syto.classification.hf_training import (
    AuxLossLoggingTrainer,
    BalancedBackgroundBatchSampler,
    BalancedTrainer,
)
# Back-compat: the old name must still import from methylbert.
from syto.classification.classifiers.methylbert import (
    MethylBertTrainer as MethylBertTrainerReexport,
    BalancedTrainer as BalancedTrainerReexport,
    BalancedBackgroundBatchSampler as SamplerReexport,
)


class TestBalancedBackgroundBatchSampler(unittest.TestCase):
    def test_even_split_covers_all_signal_and_caps_background(self):
        np.random.seed(0)
        # 8 signal (indices 0-7), 4 background (indices 8-11)
        mask = np.array([True] * 8 + [False] * 4)
        sampler = BalancedBackgroundBatchSampler(
            mask, batch_size=4, bg_ratio=0.5, shuffle=True
        )

        self.assertEqual(len(sampler), 4)  # 8 signal / 2 signal-per-batch

        batches = list(sampler)
        self.assertEqual(len(batches), 4)

        signal_seen = []
        for batch in batches:
            self.assertEqual(len(batch), 4)
            sig = [i for i in batch if i < 8]
            bg = [i for i in batch if i >= 8]
            self.assertEqual(len(sig), 2)
            self.assertEqual(len(bg), 2)
            signal_seen.extend(sig)

        # Every signal index used exactly once (even division).
        self.assertCountEqual(signal_seen, list(range(8)))

    def test_remainder_batch_emitted_when_drop_last_false(self):
        np.random.seed(0)
        mask = np.array([True] * 5 + [False] * 5)  # signal 0-4, bg 5-9
        sampler = BalancedBackgroundBatchSampler(
            mask, batch_size=4, bg_ratio=0.5, shuffle=False, drop_last=False
        )
        self.assertEqual(len(sampler), 3)  # 5 // 2 == 2 full + 1 remainder
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        # Last (remainder) batch carries the leftover signal index 4.
        self.assertIn(4, batches[-1])

    def test_drop_last_true_omits_remainder(self):
        np.random.seed(0)
        mask = np.array([True] * 5 + [False] * 5)
        sampler = BalancedBackgroundBatchSampler(
            mask, batch_size=4, bg_ratio=0.5, shuffle=False, drop_last=True
        )
        # __len__ still reports the ceil count, but iteration drops remainder.
        emitted = list(sampler)
        for batch in emitted:
            sig = [i for i in batch if i < 5]
            self.assertEqual(len(sig), 2)


class TestReexports(unittest.TestCase):
    def test_methylbert_reexports_are_the_shared_classes(self):
        self.assertIs(MethylBertTrainerReexport, AuxLossLoggingTrainer)
        self.assertIs(BalancedTrainerReexport, BalancedTrainer)
        self.assertIs(SamplerReexport, BalancedBackgroundBatchSampler)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.classification.test_hf_training -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syto.classification.hf_training'`.

- [ ] **Step 3: Create the shared module**

Create `syto/classification/hf_training.py`. Move the bodies of `BalancedBackgroundBatchSampler`, `MethylBertTrainer`, and `BalancedTrainer` verbatim from `methylbert.py` (current lines 195-337), renaming `MethylBertTrainer` → `AuxLossLoggingTrainer` and making `BalancedTrainer` extend `AuxLossLoggingTrainer`:

```python
"""Architecture-neutral HuggingFace-Trainer helpers shared across read classifiers.

These were originally written for MethylBERT but depend only on the generic HF
Trainer surface plus a boolean ``signal_mask`` over dataset indices, so any
classifier that trains through ``transformers.Trainer`` (MethylBERT, EpigenBERT)
can reuse them. Dismir trains through a hand-written loop and does NOT use these.
"""

import numpy as np
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer


class BalancedBackgroundBatchSampler(Sampler):
    """
    Yields batches where background reads are capped at bg_ratio of the batch.
    """

    def __init__(
        self, signal_mask, batch_size, bg_ratio=0.3, shuffle=True, drop_last=False
    ):
        self.batch_size = batch_size
        self.bg_ratio = bg_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.signal_indices = np.where(signal_mask)[0]
        self.bg_indices = np.where(~signal_mask)[0]

        self.n_bg_per_batch = int(batch_size * bg_ratio)
        self.n_signal_per_batch = batch_size - self.n_bg_per_batch

    def __iter__(self):
        if self.shuffle:
            signal = np.random.permutation(self.signal_indices)
            bg = np.random.permutation(self.bg_indices)
        else:
            signal = self.signal_indices.copy()
            bg = self.bg_indices.copy()

        bg_cycle = np.resize(bg, max(len(signal), len(bg) + self.batch_size))
        s_ptr, b_ptr = 0, 0

        while s_ptr + self.n_signal_per_batch <= len(signal):
            batch_signal = signal[s_ptr : s_ptr + self.n_signal_per_batch]
            batch_bg = bg_cycle[b_ptr : b_ptr + self.n_bg_per_batch]
            batch = np.concatenate([batch_signal, batch_bg])
            np.random.shuffle(batch)
            yield batch.tolist()
            s_ptr += self.n_signal_per_batch
            b_ptr += self.n_bg_per_batch

        if not self.drop_last and s_ptr < len(signal):
            remaining = signal[s_ptr:]
            n_bg_rem = int(len(remaining) * self.bg_ratio / (1 - self.bg_ratio))
            batch_bg = bg_cycle[b_ptr : b_ptr + n_bg_rem]
            batch = np.concatenate([remaining, batch_bg])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        n = len(self.signal_indices) // self.n_signal_per_batch
        if not self.drop_last and len(self.signal_indices) % self.n_signal_per_batch:
            n += 1
        return n


class AuxLossLoggingTrainer(Trainer):
    """
    Custom Trainer that optionally logs an additional loss_ce metric if provided
    by the model output (via a ``loss_ce`` attribute). Architecture-neutral: if
    the model output has no ``loss_ce``, this is a silent no-op.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_loss_ce_train = 0.0
        self._custom_loss_ce_train_steps = 0
        self._custom_loss_ce_eval = 0.0
        self._custom_loss_ce_eval_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )

        if hasattr(outputs, "loss_ce") and outputs.loss_ce is not None:
            if model.training:
                self._custom_loss_ce_train += outputs.loss_ce.item()
                self._custom_loss_ce_train_steps += 1
            else:
                self._custom_loss_ce_eval += outputs.loss_ce.item()
                self._custom_loss_ce_eval_steps += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        if "loss" in logs and getattr(self, "_custom_loss_ce_train_steps", 0) > 0:
            logs["loss_ce"] = (
                self._custom_loss_ce_train / self._custom_loss_ce_train_steps
            )
            self._custom_loss_ce_train = 0.0
            self._custom_loss_ce_train_steps = 0
        super().log(logs, *args, **kwargs)

    def evaluation_loop(self, *args, **kwargs):
        metric_key_prefix = kwargs.get("metric_key_prefix", "eval")
        if len(args) >= 5:
            metric_key_prefix = args[4]

        self._custom_loss_ce_eval = 0.0
        self._custom_loss_ce_eval_steps = 0

        output = super().evaluation_loop(*args, **kwargs)

        if (
            getattr(self, "_custom_loss_ce_eval_steps", 0) > 0
            and output.metrics is not None
        ):
            output.metrics[f"{metric_key_prefix}_loss_ce"] = (
                self._custom_loss_ce_eval / self._custom_loss_ce_eval_steps
            )
            self._custom_loss_ce_eval = 0.0
            self._custom_loss_ce_eval_steps = 0

        return output


class BalancedTrainer(AuxLossLoggingTrainer):
    """
    HF Trainer that uses BalancedBackgroundBatchSampler for training.
    """

    def __init__(self, *args, signal_mask=None, bg_ratio=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_mask = signal_mask
        self.bg_ratio = bg_ratio

    def get_train_dataloader(self) -> DataLoader:
        if self.signal_mask is None:
            # Fall back to default behavior
            return super().get_train_dataloader()

        batch_sampler = BalancedBackgroundBatchSampler(
            signal_mask=self.signal_mask,
            batch_size=self.args.per_device_train_batch_size,
            bg_ratio=self.bg_ratio,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# Back-compat alias for the pre-extraction name.
MethylBertTrainer = AuxLossLoggingTrainer
```

- [ ] **Step 4: Remove the moved classes from methylbert.py and re-export from the shared module**

In `syto/classification/classifiers/methylbert.py`, delete the three class definitions (current lines 195-337: `BalancedBackgroundBatchSampler`, `MethylBertTrainer`, `BalancedTrainer`). In their place add:

```python
from syto.classification.hf_training import (
    AuxLossLoggingTrainer,
    BalancedBackgroundBatchSampler,
    BalancedTrainer,
    MethylBertTrainer,  # back-compat alias == AuxLossLoggingTrainer
)
```

`extract_signal_mask` (current lines 107-122) and everything else in `methylbert.py` stay unchanged. `MethylBert._init_trainer` keeps referencing `MethylBertTrainer` and `BalancedTrainer` — now satisfied by the import.

- [ ] **Step 5: Run new + existing MethylBERT tests to verify green**

Run: `python -m unittest tests.classification.test_hf_training tests.classification.test_methylbert -v`
Expected: PASS (new sampler/re-export tests pass; MethylBERT suite unchanged and green).

- [ ] **Step 6: Commit**

```bash
git add syto/classification/hf_training.py tests/classification/test_hf_training.py syto/classification/classifiers/methylbert.py
git commit -m "refactor: extract shared HF balanced/aux-loss trainers into hf_training module"
```

---

### Task 2: Wire signal_mask/bg_ratio selection into EpigenDnabert2._init_trainer

**Files:**
- Modify: `syto/classification/classifiers/dnabert2.py` (imports near line 20; `_init_trainer` at lines 853-879)
- Modify: `tests/classification/test_epigenbert2.py` (`TestEpigenDnabert2TrainerBranches`, patch targets at lines 311-314, 347-350)

**Interfaces:**
- Consumes: `AuxLossLoggingTrainer`, `BalancedTrainer` from `syto.classification.hf_training`.
- Produces: `EpigenDnabert2._init_trainer(self, args=None, train_dataset=None, eval_dataset=None, model_init=None, callbacks=None, optimizers=(None, None), signal_mask=None, bg_ratio=0.3)` → returns a `BalancedTrainer` when `signal_mask is not None`, else an `AuxLossLoggingTrainer`.

- [ ] **Step 1: Write failing tests for trainer-class selection**

Add to `tests/classification/test_epigenbert2.py` inside `TestEpigenDnabert2TrainerBranches` (the `_build_stub_model` helper already exists at line 282):

```python
    def test_init_trainer_uses_aux_loss_trainer_by_default(self):
        model = self._build_stub_model(num_labels=2)
        args = MagicMock(name="training_args")
        trainer_instance = MagicMock()
        with patch(
            "syto.classification.classifiers.dnabert2.AuxLossLoggingTrainer",
            return_value=trainer_instance,
        ) as mocked_aux, patch(
            "syto.classification.classifiers.dnabert2.BalancedTrainer",
        ) as mocked_balanced:
            trainer = model._init_trainer(args=args)
        self.assertIs(trainer, trainer_instance)
        mocked_aux.assert_called_once()
        mocked_balanced.assert_not_called()

    def test_init_trainer_uses_balanced_trainer_when_signal_mask_given(self):
        import numpy as np

        model = self._build_stub_model(num_labels=2)
        args = MagicMock(name="training_args")
        mask = np.array([True, False, True, False])
        trainer_instance = MagicMock()
        with patch(
            "syto.classification.classifiers.dnabert2.BalancedTrainer",
            return_value=trainer_instance,
        ) as mocked_balanced, patch(
            "syto.classification.classifiers.dnabert2.AuxLossLoggingTrainer",
        ) as mocked_aux:
            trainer = model._init_trainer(args=args, signal_mask=mask, bg_ratio=0.4)
        self.assertIs(trainer, trainer_instance)
        mocked_aux.assert_not_called()
        kwargs = mocked_balanced.call_args.kwargs
        self.assertIs(kwargs["signal_mask"], mask)
        self.assertEqual(kwargs["bg_ratio"], 0.4)
```

Also update the two pre-existing branch tests that patch `transformers.Trainer` (lines 311-314 and 347-350): change the patch target from
`"syto.classification.classifiers.dnabert2.transformers.Trainer"` to
`"syto.classification.classifiers.dnabert2.AuxLossLoggingTrainer"`, since the default trainer class is now `AuxLossLoggingTrainer`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.classification.test_epigenbert2.TestEpigenDnabert2TrainerBranches -v`
Expected: FAIL — `AttributeError: <module 'syto...dnabert2'> does not have the attribute 'AuxLossLoggingTrainer'` (symbol not imported yet).

- [ ] **Step 3: Import the shared trainers into dnabert2**

In `syto/classification/classifiers/dnabert2.py`, near the other imports (after line 20's `from torch.utils.data import Dataset`), add:

```python
from syto.classification.hf_training import (
    AuxLossLoggingTrainer,
    BalancedTrainer,
)
```

- [ ] **Step 4: Rewrite `_init_trainer` to select the trainer class**

Replace `_init_trainer` (lines 853-879) with:

```python
    def _init_trainer(
        self,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        signal_mask=None,
        bg_ratio: float = 0.3,
    ):
        common_kwargs = dict(
            model=self.model,
            args=args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            tokenizer=self.tokenizer,
            preprocess_logits_for_metrics=preprocess_logits_for_prediction,
            compute_metrics=compute_metrics,
        )

        if signal_mask is None:
            trainer = AuxLossLoggingTrainer(**common_kwargs)
        else:
            trainer = BalancedTrainer(
                signal_mask=signal_mask,
                bg_ratio=bg_ratio,
                **common_kwargs,
            )

        use_table_progress_callback(trainer)
        return trainer
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest tests.classification.test_epigenbert2.TestEpigenDnabert2TrainerBranches -v`
Expected: PASS (new selection tests pass; the two retargeted branch tests pass).

- [ ] **Step 6: Commit**

```bash
git add syto/classification/classifiers/dnabert2.py tests/classification/test_epigenbert2.py
git commit -m "feat: select balanced/aux-loss HF trainer in EpigenDnabert2._init_trainer"
```

---

### Task 3: Thread signal_mask/bg_ratio through fine_tune and fit_classificaton

**Files:**
- Modify: `syto/classification/classifiers/dnabert2.py` (`fine_tune` at lines 960-1021; `fit_classificaton` at lines 1038-1098)
- Modify: `tests/classification/test_epigenbert2.py` (add a fine_tune-forwarding test near the existing mocked fine_tune tests, ~line 201-278)

**Interfaces:**
- Consumes: `EpigenDnabert2._init_trainer(..., signal_mask=None, bg_ratio=0.3)` from Task 2.
- Produces:
  - `EpigenDnabert2.fine_tune(..., signal_mask=None, bg_ratio=0.3)` — forwards both to `_init_trainer`.
  - `EpigenDnabert2.fit_classificaton(train_df, val_df=None, output_dir=None, **kwargs)` reads `kwargs["signal_mask"]` (default `None`) and `kwargs["bg_ratio"]` (default `0.3`) and forwards them to `fine_tune`.

- [ ] **Step 1: Write failing test that fine_tune forwards the mask to _init_trainer**

Add this test method to the **same class** that already defines `_build_stub_model()` with a mocked `_init_trainer` (the fine_tune test class at line 140, whose `_build_stub_model` mocks `_init_trainer` and `safe_save_model_for_hf_trainer` and sets `training_args = SimpleNamespace(save_model=False, output_dir=...)`). Add:

```python
    def test_fine_tune_forwards_signal_mask_and_bg_ratio(self):
        import numpy as np

        model = self._build_stub_model()  # mocks _init_trainer; save_model=False
        trainer = MagicMock(name="trainer")
        model._init_trainer.return_value = trainer
        mask = np.array([True, False, True])

        model.fine_tune(
            train_dataset=object(),
            val_dataset=object(),
            test_dataset=object(),
            signal_mask=mask,
            bg_ratio=0.25,
        )

        kwargs = model._init_trainer.call_args.kwargs
        self.assertIs(kwargs["signal_mask"], mask)
        self.assertEqual(kwargs["bg_ratio"], 0.25)
```

Note: there are two `_build_stub_model` helpers in this file (one per test class). Add this test to the fine_tune class whose helper mocks `_init_trainer` — NOT the `TestEpigenDnabert2TrainerBranches` one used in Task 2.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.classification.test_epigenbert2 -v -k fine_tune_forwards`
Expected: FAIL — `TypeError: fine_tune() got an unexpected keyword argument 'signal_mask'`.

- [ ] **Step 3: Add signal_mask/bg_ratio to fine_tune and forward to _init_trainer**

In `fine_tune` (line 960), add the two parameters to the signature (after `resume_from_checkpoint`):

```python
        resume_from_checkpoint: Optional[Union[bool, str]] = None,
        signal_mask=None,
        bg_ratio: float = 0.3,
    ):
```

Then update the `_init_trainer` call (lines 997-1002) to forward them:

```python
        self.trainer = self._init_trainer(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=self.training_args,
            callbacks=callbacks,
            signal_mask=signal_mask,
            bg_ratio=bg_ratio,
        )
```

- [ ] **Step 4: Forward signal_mask/bg_ratio from fit_classificaton**

In `fit_classificaton` (line 1038), update the `fine_tune` call (lines 1083-1092) to pass the kwargs through, mirroring how `MethylBert.fit_classificaton` does it:

```python
        self.fine_tune(
            data_path=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=val_dataset,
            training_args=self.training_args,
            callbacks=kwargs.get("callbacks", None),
            data_interface="pandas",
            resume_from_checkpoint=kwargs.get("resume_from_checkpoint", None),
            signal_mask=kwargs.get("signal_mask", None),
            bg_ratio=kwargs.get("bg_ratio", 0.3),
        )
```

- [ ] **Step 5: Run the fine_tune tests to verify pass**

Run: `python -m unittest tests.classification.test_epigenbert2 -v -k fine_tune`
Expected: PASS (new forwarding test passes; existing fine_tune tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add syto/classification/classifiers/dnabert2.py tests/classification/test_epigenbert2.py
git commit -m "feat: thread signal_mask/bg_ratio through EpigenDnabert2 fine_tune and fit_classificaton"
```

---

### Task 4: Full-suite regression check

**Files:**
- None (verification only).

- [ ] **Step 1: Run the full classification test suite**

Run: `python -m unittest discover -s tests/classification -v`
Expected: PASS — no regressions in `test_methylbert`, `test_epigenbert2`, `test_dismir`, and the new `test_hf_training`.

- [ ] **Step 2: Grep for any stale references to the moved classes' old module path**

Run: `grep -rn "class BalancedBackgroundBatchSampler\|class MethylBertTrainer\|class BalancedTrainer" syto/`
Expected: matches only in `syto/classification/hf_training.py` (the definitions), and none remaining in `methylbert.py`.

- [ ] **Step 3: Commit (only if Step 2 revealed and required a cleanup edit)**

```bash
git add -A
git commit -m "chore: verify no stale trainer/sampler class definitions remain"
```

---

## Notes / Out of Scope

- **Dismir is intentionally excluded.** It trains via a hand-written loop (`_training_loop`, `_training_loop_variable_length`) over a manual `DataLoader` with its own chunk-aware `ChunkAwareBatchSampler`. `AuxLossLoggingTrainer`/`BalancedTrainer` are HF `Trainer` subclasses with no Trainer to attach to there, and `BalancedBackgroundBatchSampler` would collide with the required chunk-aware batching. If class balance is later wanted for Dismir, implement it inside `ChunkAwareBatchSampler` as a separate, targeted change — do not extend these classes.
- **`loss_ce` is inert for DNABERT2 today.** `BertForSequenceClassification.forward` returns a stock `SequenceClassifierOutput` with no `loss_ce`, so `AuxLossLoggingTrainer`'s CE logging is a guarded no-op until/unless DNABERT2's forward is later extended to emit an auxiliary `loss_ce`. Adopting the trainer now costs nothing and unifies the two HF classifiers; the balancing benefit lands immediately.
- **Signal mask construction is the caller's job**, matching the existing `MethylBert` contract. The caller builds a boolean numpy array (length == dataset length, indexed by dataset position) — e.g. from the atlas-region annotations reads already carry — and passes it as `fit_classificaton(..., signal_mask=mask)`.
```

