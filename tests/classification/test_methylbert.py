import unittest
import tempfile
import os
from unittest.mock import MagicMock, patch
from parameterized import parameterized

from transformers import BertConfig, TrainingArguments
import torch
import pandas as pd

import syto.classification.classifiers.methylbert as methylbert_module
from syto.classification.classifiers.methylbert import (
    MethylBert,
    MethylVocab,
    MethylBertFinetuneDataset,
    MethylBertPretrainDataset,
    MethylBertEmbeddedGRG,
    VanillaClassifier,
    methylbert_finetune_collator,
    methylbert_pretrain_collator,
    _line2tokens_finetune,
    _line2tokens_pretrain,
    prepare_methylbert_list,
    _chunk_tokens,
    _generate_valid_tokens,
)


class TestChunkTokens(unittest.TestCase):
    """Tests for the token chunking helper."""

    def test_returns_original_tokens_when_sequence_is_shorter_than_window(self):
        """Returns the input as a single chunk when the read is shorter than the window."""
        tokens = ["a", "b", "c"]

        chunks = list(_chunk_tokens(tokens, window_size=5, stride=2))

        self.assertEqual(chunks, [tokens])

    def test_returns_original_tokens_when_sequence_matches_window_size(self):
        """Returns one chunk when the read length matches the window size exactly."""
        tokens = ["a", "b", "c", "d"]

        chunks = list(_chunk_tokens(tokens, window_size=4, stride=2))

        self.assertEqual(chunks, [tokens])

    def test_yields_sliding_windows_without_tail_when_last_window_already_reaches_end(
        self,
    ):
        """Avoids duplicating the final window when the sliding loop already lands on the tail."""
        tokens = list(range(9))

        chunks = list(_chunk_tokens(tokens, window_size=5, stride=2))

        self.assertEqual(chunks, [tokens[0:5], tokens[2:7], tokens[4:9]])

    def test_yields_tail_window_when_stride_skips_the_last_possible_start(self):
        """Emits the final tail window when the last valid start is missed by the stride."""
        tokens = list(range(8))

        chunks = list(_chunk_tokens(tokens, window_size=5, stride=2))

        # This is the regression case: the last possible window starts at index 3,
        # but the stride visits only starts 0 and 2, so an explicit tail chunk is required.
        self.assertEqual(chunks, [tokens[0:5], tokens[2:7], tokens[3:8]])


class TestGenerateValidTokens(unittest.TestCase):
    """Tests for k-mer generation with methylation labels."""

    def test_yields_only_kmers_without_n_bases(self):
        """Skips windows containing N and keeps the methylation code at the center base."""
        read_data = {
            "input_ids": "ATNCGA",
            "methylation_ids": "012210",
        }

        tokens = list(_generate_valid_tokens(read_data, k=3))

        self.assertEqual(tokens, [["CGA", "1"]])

    def test_returns_empty_iterator_when_no_valid_kmers_exist(self):
        """Produces no output when every candidate k-mer contains N."""
        read_data = {
            "input_ids": "NNNN",
            "methylation_ids": "0123",
        }

        tokens = list(_generate_valid_tokens(read_data, k=3))

        self.assertEqual(tokens, [])

    def test_uses_requested_kmer_size(self):
        """Supports k-mer lengths other than the default size of three."""
        read_data = {
            "input_ids": "ATCG",
            "methylation_ids": "0123",
        }

        tokens = list(_generate_valid_tokens(read_data, k=2))

        self.assertEqual(tokens, [["AT", "1"], ["TC", "2"], ["CG", "3"]])


class TestPrepareMethylbertListInference(unittest.TestCase):
    """Tests for MethylBERT inference table generation."""

    def test_builds_header_and_skips_reads_without_valid_tokens(self):
        """Keeps the header row and ignores reads that become empty after N filtering."""
        results_df = pd.DataFrame(
            [
                {
                    "read_name": "read-empty",
                    "input_ids": "NNNN",
                    "methylation_ids": "0123",
                    "grg_ctype_label": "tumor",
                    "grg_label": "grg-a",
                }
            ]
        )

        prepared = prepare_methylbert_list(
            results_df, grg_label_column="grg_label", seq_length=3, stride=1
        )

        self.assertEqual(
            prepared,
            [
                [
                    "dna_seq",
                    "methyl_seq",
                    "grg_ctype",
                    "grg_label",
                    "ctype",
                    "original_label",
                    "read_name",
                    "ncpgs_marked",
                    "on_target_mask",
                ]
            ],
        )

    def test_creates_one_output_row_per_chunk_with_metadata_propagation(self):
        """Splits valid tokens into chunks and propagates GRG metadata to each output row."""
        results_df = pd.DataFrame(
            [
                {
                    "read_name": "read-1",
                    "input_ids": "ATCGAT",
                    "methylation_ids": "010101",
                    "grg_ctype_label": "tumor",
                    "grg_label": "grg-a",
                    "original_label": 775,
                    "label": 555,
                }
            ]
        )

        prepared = prepare_methylbert_list(
            results_df,
            grg_label_column="grg_label",
            seq_length=2,
            stride=1,
            grg_ctype_label="grg_ctype_label",
        )

        self.assertEqual(len(prepared), 4)
        self.assertEqual(
            prepared[0],
            [
                "dna_seq",
                "methyl_seq",
                "grg_ctype",
                "grg_label",
                "ctype",
                "original_label",
                "read_name",
                "ncpgs_marked",
                "on_target_mask",
            ],
        )
        self.assertEqual(
            prepared[1:],
            [
                ["ATC TCG", "10", "tumor", "grg-a", 555, 775, "read-1", 2, False],
                ["TCG CGA", "01", "tumor", "grg-a", 555, 775, "read-1", 2, False],
                ["CGA GAT", "10", "tumor", "grg-a", 555, 775, "read-1", 2, False],
            ],
        )

    def test_counts_only_zero_and_one_as_marked_cpgs(self):
        """Excludes other methylation symbols from the ncpgs_marked tally."""
        results_df = pd.DataFrame(
            [
                {
                    "read_name": "read-2",
                    "input_ids": "ATCGA",
                    "methylation_ids": "12010",
                    "grg_ctype_label": "normal",
                    "grg_label": "grg-b",
                    "original_label": 775,
                    "label": 555,
                }
            ]
        )

        prepared = prepare_methylbert_list(
            results_df,
            grg_label_column="grg_label",
            seq_length=10,
            stride=5,
            grg_ctype_label="grg_ctype_label",
        )

        self.assertEqual(prepared[1][-2], 2)


class TestMethylVocab(unittest.TestCase):
    """Test suite for MethylVocab class."""

    def test_vocab_initialization(self):
        """Test basic vocabulary initialization."""
        vocab = MethylVocab(k=3)

        # Check special tokens are at expected positions
        self.assertEqual(vocab.pad_index, 0)
        self.assertEqual(vocab.unk_index, 1)
        self.assertEqual(vocab.eos_index, 2)
        self.assertEqual(vocab.sos_index, 3)
        self.assertEqual(vocab.mask_index, 4)

        # Check vocabulary contains k-mers
        self.assertIn("AAA", vocab.itos)
        self.assertIn("TTT", vocab.itos)
        self.assertIn("CCC", vocab.itos)
        self.assertIn("GGG", vocab.itos)

    def test_vocab_length(self):
        """Test vocabulary length calculation."""
        vocab = MethylVocab(k=3)

        # 5 special tokens + 4^3 = 5 + 64 = 69
        expected_length = 5 + (4**3)
        self.assertEqual(len(vocab), expected_length)

    def test_to_seq_conversion(self):
        """Test converting k-mers to sequence indices."""
        vocab = MethylVocab(k=3)

        kmers = ["AAA", "TTT", "GGG"]
        seq = vocab.to_seq(kmers)

        # Should return list of integers
        self.assertIsInstance(seq, list)
        self.assertTrue(all(isinstance(x, int) for x in seq))
        self.assertEqual(len(seq), 3)

    def test_to_seq_with_unknown(self):
        """Test handling of unknown k-mers."""
        vocab = MethylVocab(k=3)

        kmers = ["AAA", "XYZ", "TTT"]  # XYZ doesn't exist
        seq = vocab.to_seq(kmers)

        # XYZ should be converted to unk_index
        self.assertEqual(seq[1], vocab.unk_index)

    def test_from_seq_conversion(self):
        """Test converting indices back to k-mers."""
        vocab = MethylVocab(k=3)

        # Get some valid indices
        original_kmers = ["AAA", "CCC", "GGG"]
        seq = vocab.to_seq(original_kmers)

        # Convert back
        reconstructed = vocab.from_seq(seq, join=False)

        self.assertEqual(reconstructed, original_kmers)

    def test_from_seq_with_padding(self):
        """Test from_seq with padding handling."""
        vocab = MethylVocab(k=3)

        seq = [vocab.sos_index, vocab.pad_index, vocab.pad_index]

        result_with_pad = vocab.from_seq(seq, with_pad=True)
        self.assertIn("<sos>", result_with_pad)
        self.assertIn("<pad>", result_with_pad)

        # Skip padding
        result_without_pad = vocab.from_seq(seq, with_pad=False)
        self.assertIn("<sos>", result_without_pad)
        self.assertNotIn("<pad>", result_without_pad)

    def test_line2tokens_pretrain_pads_and_truncates_sequences(self):
        """Test that pretraining tokenization handles both padding and truncation."""
        truncated = _line2tokens_pretrain("AAA TTT CCC", MethylVocab(k=3), max_len=2)
        padded_vocab = MethylVocab(k=3)
        padded = _line2tokens_pretrain("AAA TTT", padded_vocab, max_len=4)

        self.assertEqual(len(truncated), 2)
        self.assertEqual(len(padded), 4)
        self.assertEqual(padded[-1], [padded_vocab.pad_index])

    def test_line2tokens_finetune_rejects_invalid_inputs(self):
        """Test that fine-tuning tokenization validates headers and field counts."""
        vocab = MethylVocab(k=3)

        with self.assertRaises(ValueError):
            _line2tokens_finetune(
                "AAA TTT\t01\t0\t1",
                tokenizer=vocab,
                max_len=5,
                headers=["dna_seq", "methyl_seq", "ctype", "grg_ctype"],
            )

        with self.assertRaises(ValueError):
            _line2tokens_finetune(
                "AAA TTT\t01\t0\t1",
                tokenizer=vocab,
                max_len=5,
                headers=[
                    "dna_seq",
                    "methyl_seq",
                    "ctype",
                    "grg_ctype",
                    "grg_label",
                ],
            )


class TestMethylBertFinetuneDataset(unittest.TestCase):
    """Test suite for MethylBertFinetuneDataset."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab = MethylVocab(k=3)
        self.seq_len = 150

    def test_dataset_from_list(self):
        """Test dataset creation from list of lists."""
        # Create example data
        data = [
            ["dna_seq", "methyl_seq", "ctype", "grg_ctype", "grg_label"],
            ["AAA TTT CCC", "012", "0", "0", "0"],
            ["GGG AAA TTT", "120", "1", "0", "1"],
        ]

        dataset = MethylBertFinetuneDataset(
            data_source=data, vocab=self.vocab, seq_len=self.seq_len, n_cores=1
        )

        self.assertEqual(len(dataset), 2)

    def test_dataset_getitem(self):
        """Test retrieving items from dataset."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "grg_ctype", "grg_label"],
            ["AAA TTT CCC", "012", "0", "1", "0"],
        ]

        dataset = MethylBertFinetuneDataset(
            data_source=data, vocab=self.vocab, seq_len=self.seq_len, n_cores=1
        )

        item = dataset[0]

        # Check structure
        self.assertIn("input_ids", item)
        self.assertIn("token_type_ids", item)
        self.assertIn("labels", item)
        self.assertIn("grg_ids", item)

        # Check types
        self.assertIsInstance(item["input_ids"], torch.Tensor)
        self.assertIsInstance(item["token_type_ids"], torch.Tensor)
        self.assertIsInstance(item["labels"], int)
        self.assertIsInstance(item["grg_ids"], int)

    def test_dataset_adds_missing_grg_label(self):
        """Test that dataset adds default grg_label if missing."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "grg_ctype"],
            ["AAA TTT", "01", "0", "1"],
        ]

        dataset = MethylBertFinetuneDataset(
            data_source=data, vocab=self.vocab, seq_len=self.seq_len, n_cores=1
        )

        item = dataset[0]
        self.assertEqual(item["grg_ids"], 0)  # Default value

    def test_dataset_special_tokens(self):
        """Test that SOS and EOS tokens are added."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "grg_ctype", "grg_label"],
            ["AAA TTT", "01", "1", "0", "0"],
        ]

        dataset = MethylBertFinetuneDataset(
            data_source=data, vocab=self.vocab, seq_len=self.seq_len, n_cores=1
        )

        item = dataset[0]

        # First token should be SOS
        self.assertEqual(item["input_ids"][0].item(), self.vocab.sos_index)

        # Should have EOS somewhere
        self.assertIn(self.vocab.eos_index, item["input_ids"].tolist())

    def test_dataset_num_grs(self):
        """Test num_grs method."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "grg_ctype", "grg_label"],
            ["AAA", "0", "1", "1", "0"],
            ["TTT", "1", "2", "1", "1"],
            ["CCC", "2", "2", "1", "2"],
        ]

        dataset = MethylBertFinetuneDataset(
            data_source=data, vocab=self.vocab, seq_len=self.seq_len, n_cores=1
        )

        num_grs = dataset.num_grs()
        self.assertGreaterEqual(num_grs, 3)  # At least 3 unique GR labels

    def test_lazy_tokenization_caches_items_and_num_grs(self):
        """Test lazy tokenization cache population and on-demand GR counting."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "grg_ctype", "grg_label"],
            ["AAA TTT CCC GGG", "0120", "0", "1", "0"],
            ["GGG CCC AAA TTT", "1201", "1", "1", "2"],
        ]

        with tempfile.TemporaryDirectory() as cache_dir:
            dataset = MethylBertFinetuneDataset(
                data_source=data,
                vocab=self.vocab,
                seq_len=4,
                n_cores=1,
                lazy_tokenization=True,
                cache_dir=cache_dir,
            )

            _ = dataset[0]  # Trigger tokenization and caching of first item

            self.assertIn(0, dataset._cache)
            self.assertEqual(dataset.num_grs(), 3)

    def test_eager_mode_reuses_cache_file(self):
        """Test that eager mode reloads tokenized content from cache."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "grg_ctype", "grg_label"],
            ["AAA TTT CCC GGG", "0120", "0", "1", "0"],
            ["GGG CCC AAA TTT", "1201", "1", "1", "2"],
        ]

        with tempfile.TemporaryDirectory() as cache_dir:
            dataset = MethylBertFinetuneDataset(
                data_source=data,
                vocab=self.vocab,
                seq_len=4,
                n_cores=1,
                cache_dir=cache_dir,
            )

            self.assertTrue(os.path.exists(dataset.cache_file))

            with patch.object(
                methylbert_module,
                "_line2tokens_finetune",
                side_effect=AssertionError("cache was not reused"),
            ):
                cached_dataset = MethylBertFinetuneDataset(
                    data_source=data,
                    vocab=self.vocab,
                    seq_len=4,
                    n_cores=1,
                    cache_dir=cache_dir,
                )

            self.assertEqual(len(cached_dataset), 2)


class TestMethylBertCollators(unittest.TestCase):
    """Test suite for data collators."""

    def test_finetune_collator(self):
        """Test finetune data collator."""
        features = [
            {
                "input_ids": torch.randint(0, 10, (150,)),
                "token_type_ids": torch.randint(0, 3, (150,)),
                "labels": 0,
                "grg_ids": 1,
            },
            {
                "input_ids": torch.randint(0, 10, (150,)),
                "token_type_ids": torch.randint(0, 3, (150,)),
                "labels": 1,
                "grg_ids": 2,
            },
        ]

        batch = methylbert_finetune_collator(features)

        # Check batch structure
        self.assertIn("input_ids", batch)
        self.assertIn("token_type_ids", batch)
        self.assertIn("labels", batch)
        self.assertIn("grg_ids", batch)

        # Check shapes
        self.assertEqual(batch["input_ids"].shape[0], 2)  # Batch size
        self.assertEqual(batch["token_type_ids"].shape[0], 2)
        self.assertEqual(batch["labels"].shape[0], 2)
        self.assertEqual(batch["grg_ids"].shape[0], 2)

    def test_pretrain_collator(self):
        """Test pretrain data collator."""
        features = [
            {
                "bert_input": torch.randint(0, 10, (150,)),
                "bert_label": torch.randint(0, 10, (150,)),
                "bert_mask": torch.randint(0, 2, (150,)).bool(),
            },
            {
                "bert_input": torch.randint(0, 10, (150,)),
                "bert_label": torch.randint(0, 10, (150,)),
                "bert_mask": torch.randint(0, 2, (150,)).bool(),
            },
        ]

        batch = methylbert_pretrain_collator(features)

        # Check batch structure
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertIn("bert_mask", batch)

        # Check shapes
        self.assertEqual(batch["input_ids"].shape[0], 2)


class TestMethylBert(unittest.TestCase):
    """Test suite for MethylBert main class."""

    def setUp(self):
        """Set up test fixtures."""
        self.foundation_model = "foundationalModels/methylbert_hg19_12l"
        self.seq_len = 150
        self.loss = "bce"

    def _build_small_hf_config(self, num_labels=2, num_grg_labels=4, loss="bce"):
        """Build a tiny BERT config so wrapper-level tests stay lightweight."""
        config = BertConfig(
            vocab_size=80,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=24,
            max_position_embeddings=32,
            type_vocab_size=3,
        )
        config.num_labels = num_labels
        config.num_grg_labels = num_grg_labels
        config.loss = loss
        return config

    def _create_small_model(self, **kwargs):
        """Create a wrapper with a tiny Hugging Face config for unit-level branches."""
        loss = kwargs.get("loss", self.loss)
        with patch(
            "syto.classification.classifiers.methylbert.BertConfig.from_pretrained",
            return_value=self._build_small_hf_config(
                num_labels=kwargs.get("num_labels", 2),
                num_grg_labels=kwargs.get("num_grg_labels", 4),
                loss=loss,
            ),
        ), patch(
            "syto.classification.classifiers.methylbert.AutoTokenizer.from_pretrained",
            return_value=None,
        ):
            return MethylBert(
                foundation_model_path=self.foundation_model,
                seq_len=kwargs.pop("seq_len", 5),
                loss=kwargs.pop("loss", self.loss),
                load_weights=kwargs.pop("load_weights", False),
                num_labels=kwargs.pop("num_labels", 2),
                num_grg_labels=kwargs.pop("num_grg_labels", 4),
                output_dir=kwargs.pop(
                    "output_dir", "../test_container_tmp/tmp_trainer"
                ),
                **kwargs,
            )

    def _create_model(
        self, num_labels=2, num_grg_labels=10, load_weights=False, loss=None
    ):
        """Helper to create a MethylBert model."""
        return MethylBert(
            foundation_model_path=self.foundation_model,
            seq_len=self.seq_len,
            loss=self.loss if loss is None else loss,
            load_weights=load_weights,
            num_labels=num_labels,
            num_grg_labels=num_grg_labels,
            output_dir="../test_container_tmp/tmp_trainer",
        )

    def test_model_initialization_without_weights(self):
        """Test model initialization without loading weights."""
        model = self._create_model(load_weights=False)

        self.assertIsNotNone(model.model)
        self.assertEqual(model.seq_len, self.seq_len)
        self.assertIsNotNone(model.training_args)

    def test_model_initialization_with_different_labels(self):
        """Test model initialization with different number of labels."""
        model_binary = self._create_model(num_labels=2, load_weights=False)
        model_multiclass = self._create_model(num_labels=5, load_weights=False)

        self.assertEqual(model_binary.model.num_labels, 2)
        self.assertEqual(model_multiclass.model.num_labels, 5)

    def test_model_config_propagation(self):
        """Test that config is properly propagated to model."""
        training_args = TrainingArguments(
            output_dir="../test_container_tmp/tmp_trainer",
            learning_rate=0.001,
        )

        model = MethylBert(
            foundation_model_path=self.foundation_model,
            seq_len=self.seq_len,
            training_args=training_args,
            load_weights=False,
            num_labels=2,
            num_grg_labels=10,
            output_dir="../test_container_tmp/tmp_trainer",
        )

        self.assertEqual(model.training_args.learning_rate, 0.001)

    @parameterized.expand(
        [
            ("binary", 2, 10),
            ("multiclass", 5, 20),
        ]
    )
    def test_model_predict(self, name, num_labels, num_grg_labels):
        """Test model prediction with different configurations."""
        loss = "ce" if num_labels > 2 else "bce"
        model = self._create_model(
            num_labels=num_labels,
            num_grg_labels=num_grg_labels,
            load_weights=False,
            loss=loss,
        )

        # Create dummy dataset
        vocab = MethylVocab(k=3)
        data = [
            ["dna_seq", "methyl_seq", "ctype", "grg_ctype", "grg_label"],
            ["AAA TTT CCC GGG", "0120", 0, 0, 1],
            ["GGG CCC AAA TTT", "1201", 1, 1, 2],
        ]

        dataset = MethylBertFinetuneDataset(
            data_source=data, vocab=vocab, seq_len=self.seq_len, n_cores=1
        )

        # Predict
        predictions = model.predict(dataset, batch_size=2)

        self.assertIsNotNone(predictions)
        self.assertIsNotNone(predictions.predictions)
        self.assertEqual(len(predictions.predictions), 2)

    def test_vanilla_classifier_forward_returns_expected_shapes(self):
        """Test that the standalone vanilla classifier produces logits and augmented features."""
        config = self._build_small_hf_config(num_labels=3)
        classifier = VanillaClassifier(config, seq_len=5)
        sequence_output = torch.randn(2, 6, 12)
        grg_ids = torch.tensor([1, 2], dtype=torch.long)

        logits, sequence_output_with_gr = classifier(sequence_output, grg_ids)

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(sequence_output_with_gr.shape, (2, 6, 13))

    def test_embedded_model_rejects_invalid_loss(self):
        """Test that the embedded model validates unknown losses."""
        config = self._build_small_hf_config(loss="not_a_loss")

        with self.assertRaises(ValueError):
            MethylBertEmbeddedGRG(config, seq_len=5)

    def test_embedded_model_forward_attention_classifier_returns_attention_weights(
        self,
    ):
        """Test the attention-based classifier forward path on a tiny configuration."""
        config = self._build_small_hf_config(num_labels=3, loss="ce")
        model = MethylBertEmbeddedGRG(
            config,
            seq_len=5,
            classifier_implementation="grg_attention_based",
        )
        input_ids = torch.randint(5, 20, (2, 6))
        token_type_ids = torch.randint(0, 3, (2, 6))
        attention_mask = torch.ones(2, 6, dtype=torch.long)
        attention_mask[1, -1] = 0

        # Masking one position forces the attention branch to use its mask logic.
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=torch.tensor([1, 2], dtype=torch.long),
            grg_ids=torch.tensor([1, 2], dtype=torch.long),
        )

        self.assertIsNotNone(output.loss)
        self.assertEqual(output.logits.shape, (2, 3))
        self.assertEqual(output.attention_weights.shape, (2, 6))

    def test_wrapper_rejects_invalid_classifier_implementation(self):
        """Test that wrapper initialization validates classifier implementation names."""
        with self.assertRaises(ValueError):
            MethylBert(
                foundation_model_path=self.foundation_model,
                seq_len=5,
                loss=self.loss,
                load_weights=False,
                classifier_implementation="invalid-name",
            )

    def test_wrapper_loads_finetuned_checkpoint_path(self):
        """Test that loading weights with a fine-tuned path uses that checkpoint."""
        mocked_model = MagicMock()

        with patch(
            "syto.classification.classifiers.methylbert.BertConfig.from_pretrained",
            return_value=self._build_small_hf_config(),
        ), patch(
            "syto.classification.classifiers.methylbert.MethylBertEmbeddedGRG.from_pretrained",
            return_value=mocked_model,
        ) as mocked_from_pretrained, patch(
            "syto.classification.classifiers.methylbert.AutoTokenizer.from_pretrained",
            return_value=object(),
        ):
            model = MethylBert(
                foundation_model_path=self.foundation_model,
                seq_len=5,
                loss=self.loss,
                load_weights=True,
                fine_tuned_model_path="/tmp/fine_tuned_model",
                num_labels=2,
                num_grg_labels=4,
            )

        self.assertIs(model.model, mocked_model)
        self.assertEqual(
            mocked_from_pretrained.call_args.kwargs["pretrained_model_name_or_path"],
            "/tmp/fine_tuned_model",
        )

    def test_predict_reuses_existing_trainer_and_clears_cache(self):
        """Test that prediction reuses the existing trainer and clears caches."""
        model = self._create_small_model()
        model.trainer = MagicMock()
        sentinel_predictions = object()
        model.trainer.predict.return_value = sentinel_predictions

        with patch("gc.collect") as mocked_collect, patch(
            "torch.cuda.empty_cache"
        ) as mocked_empty_cache:
            predictions = model.predict(dataset=[{"input_ids": torch.tensor([1])}])

        self.assertIs(predictions, sentinel_predictions)
        mocked_collect.assert_called_once()
        mocked_empty_cache.assert_called_once()


class TestMethylBertFineTune(unittest.TestCase):
    """Test suite for MethylBert fine-tuning."""

    def setUp(self):
        """Set up test fixtures."""
        self.foundation_model = "foundationalModels/methylbert_hg19_12l"
        self.seq_len = 150
        self.loss = "bce"
        self.vocab = MethylVocab(k=3)

    def _create_test_dataset(self, num_samples=5):
        """Helper to create a test dataset using generate_example_data_for_methylbert."""
        from syto.data.dataset import generate_example_data_for_methylbert

        data = generate_example_data_for_methylbert(
            sequence_length=self.seq_len,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=num_samples,
        )

        return MethylBertFinetuneDataset(
            data_source=data, vocab=self.vocab, seq_len=self.seq_len, n_cores=1
        )

    def test_fine_tune_basic(self):
        """Test basic fine-tuning functionality."""
        model = MethylBert(
            foundation_model_path=self.foundation_model,
            seq_len=self.seq_len,
            loss=self.loss,
            load_weights=False,
            num_labels=2,
            num_grg_labels=10,
            output_dir="../test_container_tmp/tmp_trainer",
        )

        train_dataset = self._create_test_dataset(10)
        val_dataset = self._create_test_dataset(5)

        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = TrainingArguments(
                output_dir=temp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                eval_strategy="steps",
                eval_steps=2,
                save_steps=2,
                logging_steps=1,
                save_total_limit=1,
                load_best_model_at_end=False,
                report_to=[],
            )

            # This should run without errors
            model.fine_tune(
                data_path=None,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                training_args=training_args,
            )

            self.assertIsNotNone(model.trainer)

    def test_splitting_long_sequences(self):
        """Test fine-tuning with sequences longer than 512."""
        model = MethylBert(
            foundation_model_path=self.foundation_model,
            seq_len=600,  # Longer than 512
            loss=self.loss,
            load_weights=False,
            num_labels=2,
            num_grg_labels=10,
            output_dir="../test_container_tmp/tmp_trainer",
        )

        from syto.data.dataset import generate_example_data_for_methylbert

        # generate_example_data_for_methylbert should handle splitting
        data = generate_example_data_for_methylbert(
            sequence_length=600,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=5,
        )

        dataset = MethylBertFinetuneDataset(
            data_source=data, vocab=self.vocab, seq_len=600, n_cores=1
        )

        # Should handle without errors
        self.assertGreater(len(dataset), 5)  # More samples due to splitting


class TestMethylBertPretrainDataset(unittest.TestCase):
    """Test suite for MethylBertPretrainDataset."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab = MethylVocab(k=3)
        self.seq_len = 120

    def test_pretrain_dataset_from_file(self):
        """Test pretrain dataset creation from file."""
        from syto.data.dataset import generate_example_data_for_methylbert

        # Generate synthetic data
        data = generate_example_data_for_methylbert(
            sequence_length=self.seq_len,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=3,
        )

        # Write to temp file (only dna_seq column for pretrain)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            # Skip header, write only dna sequences
            for row in data[1:]:
                f.write(row[0] + "\n")  # dna_seq column
            temp_file = f.name

        try:
            dataset = MethylBertPretrainDataset(
                f_path=temp_file, vocab=self.vocab, seq_len=self.seq_len
            )

            self.assertEqual(len(dataset), 3)

            # Test getitem
            item = dataset[0]
            self.assertIn("bert_input", item)
            self.assertIn("bert_label", item)
            self.assertIn("bert_mask", item)

        finally:
            os.unlink(temp_file)

    @parameterized.expand(
        [
            (1),
            (5),
        ]
    )
    def test_pretrain_dataset_masking(self, num_samples):
        """Test that masking is applied in pretrain dataset."""
        from syto.data.dataset import generate_example_data_for_methylbert

        # Generate a longer sequence to ensure masking
        data = generate_example_data_for_methylbert(
            sequence_length=self.seq_len,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=num_samples,
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            for row in data[1:]:
                f.write(row[0] + "\n")  # dna_seq column
            temp_file = f.name

        try:
            dataset = MethylBertPretrainDataset(
                f_path=temp_file, vocab=self.vocab, seq_len=self.seq_len
            )

            item = dataset[0]

            # Check that some tokens are masked
            self.assertIn(self.vocab.mask_index, item["bert_input"].tolist())

            # Check that label has -100 for unmasked positions
            self.assertIn(-100, item["bert_label"].tolist())

        finally:
            os.unlink(temp_file)

    def test_pretrain_dataset_with_various_lengths(self):
        """Test pretrain dataset handles various sequence lengths."""
        from syto.data.dataset import generate_example_data_for_methylbert

        for seq_len in [50, 120, 200]:
            with self.subTest(seq_len=seq_len):
                data = generate_example_data_for_methylbert(
                    sequence_length=seq_len,
                    include_cpg_methylation=True,
                    include_labels=True,
                    num_samples=2,
                )

                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".txt"
                ) as f:
                    for row in data[1:]:
                        f.write(row[0] + "\n")
                    temp_file = f.name

                try:
                    dataset = MethylBertPretrainDataset(
                        f_path=temp_file, vocab=self.vocab, seq_len=seq_len
                    )

                    self.assertGreater(len(dataset), 0)

                    item = dataset[0]
                    # Check tensors have expected shape
                    self.assertEqual(item["bert_input"].shape[0], seq_len + 1)

                finally:
                    os.unlink(temp_file)

    def test_masking_adds_special_tokens_and_marks_masked_positions(self):
        """Test the masking helper end-to-end on a dense, fully maskable sequence."""
        dataset = object.__new__(MethylBertPretrainDataset)
        dataset.vocab = self.vocab
        dataset.seq_len = 6
        dataset.mask_list = dataset._get_mask()
        inputs = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.int16)

        masked_inputs, labels, masked_positions = dataset._masking(
            inputs.clone(), threshold=1.0
        )

        # With threshold=1.0 every non-special token is selected, so the helper should
        # prepend SOS and align labels and mask flags with the shifted sequence.
        self.assertEqual(masked_inputs[0].item(), self.vocab.sos_index)
        self.assertEqual(masked_inputs[-1].item(), self.vocab.eos_index)
        self.assertEqual(labels[0].item(), -100)
        self.assertFalse(masked_positions[0].item())
        self.assertEqual(masked_inputs.shape[0], 7)

    def test_random_length_branch_keeps_output_shape(self):
        """Test that random-length truncation still returns padded tensors of the expected size."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("AAA TTT CCC GGG AAA TTT\n")
            temp_file = f.name

        try:
            dataset = MethylBertPretrainDataset(
                f_path=temp_file,
                vocab=self.vocab,
                seq_len=6,
                random_len=True,
                n_cores=1,
            )

            with patch("numpy.random.random", return_value=0.0), patch(
                "random.randint", return_value=5
            ):
                item = dataset[0]

            self.assertEqual(item["bert_input"].shape[0], 7)
            self.assertEqual(item["bert_label"].shape[0], 7)
            self.assertEqual(item["bert_mask"].shape[0], 7)
        finally:
            os.unlink(temp_file)


class TestMethylBertSoftCollator(unittest.TestCase):
    """Test suite for methylbert_finetune_soft_collator."""

    def test_soft_finetune_collator(self):
        """Test soft-label collator stacks float labels and on_target_mask."""
        from syto.classification.classifiers.methylbert import (
            methylbert_finetune_collator,
        )

        num_classes = 5
        features = [
            {
                "input_ids": torch.randint(0, 10, (150,)),
                "token_type_ids": torch.randint(0, 3, (150,)),
                "labels": torch.softmax(torch.randn(num_classes), dim=-1),
                "grg_ids": 1,
                "on_target_mask": torch.tensor(True),
            },
            {
                "input_ids": torch.randint(0, 10, (150,)),
                "token_type_ids": torch.randint(0, 3, (150,)),
                "labels": torch.softmax(torch.randn(num_classes), dim=-1),
                "grg_ids": 2,
                "on_target_mask": torch.tensor(False),
            },
        ]

        batch = methylbert_finetune_collator(features)

        # Check batch structure
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertIn("grg_ids", batch)
        self.assertIn("on_target_mask", batch)

        # Labels should be float tensors of shape [batch_size, num_classes]
        self.assertEqual(batch["labels"].shape, (2, num_classes))
        self.assertEqual(batch["labels"].dtype, torch.float32)

        # grg_ids should be long
        self.assertEqual(batch["grg_ids"].dtype, torch.long)

    def test_soft_finetune_collator_with_list_labels(self):
        """Test that collator handles list labels (non-tensor) too."""
        from syto.classification.classifiers.methylbert import (
            methylbert_finetune_collator,
        )

        features = [
            {
                "input_ids": torch.randint(0, 10, (150,)),
                "token_type_ids": torch.randint(0, 3, (150,)),
                "labels": [0.8, 0.1, 0.1],
                "grg_ids": 0,
                "on_target_mask": torch.tensor(True),
            },
        ]

        batch = methylbert_finetune_collator(features)
        self.assertEqual(batch["labels"].shape, (1, 3))
        self.assertEqual(batch["labels"].dtype, torch.float32)


class TestMethylBertSoftLabelLossSetup(unittest.TestCase):
    """Test loss setup for soft-label loss types."""

    def _build_small_hf_config(self, num_labels=5, loss="cwce"):
        config = BertConfig(
            vocab_size=80,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=24,
            max_position_embeddings=32,
            type_vocab_size=3,
        )
        config.num_labels = num_labels
        config.num_grg_labels = 4
        config.loss = loss
        return config

    def test_setup_loss_cwce(self):
        """Verify _setup_loss('cwce') returns ConfidenceWeightedCrossEntropy."""
        from syto.classification.loss import ConfidenceWeightedCrossEntropy

        config = self._build_small_hf_config(num_labels=5, loss="cwce")
        model = MethylBertEmbeddedGRG(config, seq_len=5)
        self.assertIsInstance(
            model.classification_loss_fct, ConfidenceWeightedCrossEntropy
        )


class TestMethylBertSoftLabelForward(unittest.TestCase):
    """Test forward pass with soft labels."""

    def _build_small_hf_config(self, num_labels=5, loss="cwce", on_target_weight=None):
        config = BertConfig(
            vocab_size=80,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=3,
            intermediate_size=24,
            max_position_embeddings=32,
            type_vocab_size=3,
        )
        config.num_labels = num_labels
        config.num_grg_labels = 4
        config.loss = loss
        config.on_target_weight = on_target_weight
        return config

    def test_forward_with_soft_labels_cwce(self):
        """Forward pass with loss='cwce' and 2D float soft labels should compute loss."""
        config = self._build_small_hf_config(num_labels=5, loss="cwce")
        model = MethylBertEmbeddedGRG(config, seq_len=5)

        input_ids = torch.randint(5, 20, (2, 6))
        token_type_ids = torch.randint(0, 3, (2, 6))
        # Soft labels: [batch_size, num_classes]
        soft_labels = torch.softmax(torch.randn(2, 5), dim=-1)
        grg_ids = torch.tensor([1, 2], dtype=torch.long)

        output = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            labels=soft_labels,
            grg_ids=grg_ids,
        )

        self.assertIsNotNone(output.loss)
        self.assertFalse(torch.isnan(output.loss))
        self.assertEqual(output.logits.shape, (2, 5))

    def test_forward_with_soft_labels_cwce_with_on_target_weight(self):
        """Forward pass with loss='cwce', soft labels, and on_target_mask."""
        config = self._build_small_hf_config(
            num_labels=5, loss="cwce", on_target_weight=1.0
        )
        model = MethylBertEmbeddedGRG(config, seq_len=5)

        input_ids = torch.randint(5, 20, (2, 6))
        token_type_ids = torch.randint(0, 3, (2, 6))
        soft_labels = torch.softmax(torch.randn(2, 5), dim=-1)
        grg_ids = torch.tensor([1, 2], dtype=torch.long)
        on_target_mask = torch.tensor([True, False])

        output = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            labels=soft_labels,
            grg_ids=grg_ids,
            on_target_mask=on_target_mask,
        )

        self.assertIsNotNone(output.loss)
        self.assertFalse(torch.isnan(output.loss))
        self.assertEqual(output.logits.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
