import unittest
import tempfile
import torch
import os
from parameterized import parameterized
from methyldl.modelling.methylbert import (
    MethylBert,
    MethylVocab,
    MethylBertFinetuneDataset,
    MethylBertPretrainDataset,
    methylbert_finetune_collator,
    methylbert_pretrain_collator,
    default_methylbert_config,
    FocalLoss,
    sigmoid_focal_loss
)
from transformers import TrainingArguments
from methyldl.data.dataset import generate_example_data_for_methylbert
from methyldl.data.dataset import generate_example_data

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
        expected_length = 5 + (4 ** 3)
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


class TestFocalLoss(unittest.TestCase):
    """Test suite for Focal Loss implementation."""
    
    def test_focal_loss_basic(self):
        """Test basic focal loss computation."""
        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10, 2)).float()
        
        loss = sigmoid_focal_loss(inputs, targets, reduction='mean')
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_focal_loss_reduction_modes(self):
        """Test different reduction modes."""
        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10, 2)).float()
        
        loss_none = sigmoid_focal_loss(inputs, targets, reduction='none')
        loss_mean = sigmoid_focal_loss(inputs, targets, reduction='mean')
        loss_sum = sigmoid_focal_loss(inputs, targets, reduction='sum')
        
        self.assertEqual(loss_none.shape, inputs.shape)
        self.assertEqual(loss_mean.shape, torch.Size([]))
        self.assertEqual(loss_sum.shape, torch.Size([]))
    
    def test_focal_loss_class(self):
        """Test FocalLoss class wrapper."""
        criterion = FocalLoss(reduction='mean')
        
        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10, 2)).float()
        
        loss = criterion(inputs, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)


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
            ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"],
            ["AAA TTT CCC", "012", "type1", "type1", "0"],
            ["GGG AAA TTT", "120", "type1", "type2", "1"]
        ]
        
        dataset = MethylBertFinetuneDataset(
            data_source=data,
            vocab=self.vocab,
            seq_len=self.seq_len,
            n_cores=1
        )
        
        self.assertEqual(len(dataset), 2)
    
    def test_dataset_getitem(self):
        """Test retrieving items from dataset."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"],
            ["AAA TTT CCC", "012", "type1", "type1", "0"]
        ]
        
        dataset = MethylBertFinetuneDataset(
            data_source=data,
            vocab=self.vocab,
            seq_len=self.seq_len,
            n_cores=1
        )
        
        item = dataset[0]
        
        # Check structure
        self.assertIn("input_ids", item)
        self.assertIn("token_type_ids", item)
        self.assertIn("labels", item)
        self.assertIn("dmr_ids", item)
        
        # Check types
        self.assertIsInstance(item["input_ids"], torch.Tensor)
        self.assertIsInstance(item["token_type_ids"], torch.Tensor)
        self.assertIsInstance(item["labels"], int)
        self.assertIsInstance(item["dmr_ids"], int)
    
    def test_dataset_adds_missing_dmr_label(self):
        """Test that dataset adds default dmr_label if missing."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "dmr_ctype"],
            ["AAA TTT", "01", "type1", "type1"]
        ]
        
        dataset = MethylBertFinetuneDataset(
            data_source=data,
            vocab=self.vocab,
            seq_len=self.seq_len,
            n_cores=1
        )
        
        item = dataset[0]
        self.assertEqual(item["dmr_ids"], 0)  # Default value
    
    
    def test_dataset_special_tokens(self):
        """Test that SOS and EOS tokens are added."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"],
            ["AAA TTT", "01", "type1", "type1", "0"]
        ]
        
        dataset = MethylBertFinetuneDataset(
            data_source=data,
            vocab=self.vocab,
            seq_len=self.seq_len,
            n_cores=1
        )
        
        item = dataset[0]
        
        # First token should be SOS
        self.assertEqual(item["input_ids"][0].item(), self.vocab.sos_index)
        
        # Should have EOS somewhere
        self.assertIn(self.vocab.eos_index, item["input_ids"].tolist())
    
    def test_dataset_num_dmrs(self):
        """Test num_dmrs method."""
        data = [
            ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"],
            ["AAA", "0", "type1", "type1", "0"],
            ["TTT", "1", "type1", "type2", "1"],
            ["CCC", "2", "type2", "type2", "2"]
        ]
        
        dataset = MethylBertFinetuneDataset(
            data_source=data,
            vocab=self.vocab,
            seq_len=self.seq_len,
            n_cores=1
        )
        
        num_dmrs = dataset.num_dmrs()
        self.assertGreaterEqual(num_dmrs, 3)  # At least 3 unique DMR labels


class TestMethylBertCollators(unittest.TestCase):
    """Test suite for data collators."""
    
    def test_finetune_collator(self):
        """Test finetune data collator."""
        features = [
            {
                "input_ids": torch.randint(0, 10, (150,)),
                "token_type_ids": torch.randint(0, 3, (150,)),
                "labels": 0,
                "dmr_ids": 1
            },
            {
                "input_ids": torch.randint(0, 10, (150,)),
                "token_type_ids": torch.randint(0, 3, (150,)),
                "labels": 1,
                "dmr_ids": 2
            }
        ]
        
        batch = methylbert_finetune_collator(features)
        
        # Check batch structure
        self.assertIn("input_ids", batch)
        self.assertIn("token_type_ids", batch)
        self.assertIn("labels", batch)
        self.assertIn("dmr_ids", batch)
        
        # Check shapes
        self.assertEqual(batch["input_ids"].shape[0], 2)  # Batch size
        self.assertEqual(batch["token_type_ids"].shape[0], 2)
        self.assertEqual(batch["labels"].shape[0], 2)
        self.assertEqual(batch["dmr_ids"].shape[0], 2)
    
    def test_pretrain_collator(self):
        """Test pretrain data collator."""
        features = [
            {
                "bert_input": torch.randint(0, 10, (150,)),
                "bert_label": torch.randint(0, 10, (150,)),
                "bert_mask": torch.randint(0, 2, (150,)).bool()
            },
            {
                "bert_input": torch.randint(0, 10, (150,)),
                "bert_label": torch.randint(0, 10, (150,)),
                "bert_mask": torch.randint(0, 2, (150,)).bool()
            }
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
        self.foundation_model = "hanyangii/methylbert_hg19_2l"
        self.seq_len = 150
        self.config = default_methylbert_config.copy()
    
    def _create_model(self, num_labels=2, num_dmr_labels=10, load_weights=False):
        """Helper to create a MethylBert model."""
        return MethylBert(
            foundation_model_path=self.foundation_model,
            seq_len=self.seq_len,
            custom_config=self.config,
            load_weights=load_weights,
            num_labels=num_labels,
            num_dmr_labels=num_dmr_labels
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
        custom_config = default_methylbert_config.copy()
        custom_config["lr"] = 0.001
        
        model = MethylBert(
            foundation_model_path=self.foundation_model,
            seq_len=self.seq_len,
            custom_config=custom_config,
            load_weights=False,
            num_labels=2,
            num_dmr_labels=10
        )
        
        self.assertEqual(model.training_args.learning_rate, 0.001)
    
    @parameterized.expand([
        ("binary", 2, 10),
        ("multiclass", 5, 20),
    ])
    def test_model_predict(self, name, num_labels, num_dmr_labels):
        """Test model prediction with different configurations."""
        model = self._create_model(
            num_labels=num_labels,
            num_dmr_labels=num_dmr_labels,
            load_weights=False
        )
        
        # Create dummy dataset
        vocab = MethylVocab(k=3)
        data = [
            ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"],
            ["AAA TTT CCC GGG", "0120", "type1", "type1", "0"],
            ["GGG CCC AAA TTT", "1201", "type1", "type2", "1"]
        ]
        
        dataset = MethylBertFinetuneDataset(
            data_source=data,
            vocab=vocab,
            seq_len=self.seq_len,
            n_cores=1
        )
        
        # Predict
        predictions = model.predict(dataset, batch_size=2)
        
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(predictions.predictions)
        self.assertEqual(len(predictions.predictions), 2)


class TestMethylBertFineTune(unittest.TestCase):
    """Test suite for MethylBert fine-tuning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.foundation_model = "hanyangii/methylbert_hg19_2l"
        self.seq_len = 150
        self.config = default_methylbert_config.copy()
        self.vocab = MethylVocab(k=3)
    
    def _create_test_dataset(self, num_samples=5):
        """Helper to create a test dataset using generate_example_data_for_methylbert."""
        from methyldl.data.dataset import generate_example_data_for_methylbert
        
        data = generate_example_data_for_methylbert(
            sequence_length=self.seq_len,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=num_samples
        )

        return MethylBertFinetuneDataset(
            data_source=data,
            vocab=self.vocab,
            seq_len=self.seq_len,
            n_cores=1
        )
    

    
    def test_fine_tune_basic(self):
        """Test basic fine-tuning functionality."""
        model = MethylBert(
            foundation_model_path=self.foundation_model,
            seq_len=self.seq_len,
            custom_config=self.config,
            load_weights=False,
            num_labels=2,
            num_dmr_labels=10
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
                load_best_model_at_end=False
            )
            
            # This should run without errors
            model.fine_tune(
                data_path=None,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                training_args=training_args
            )
            
            self.assertIsNotNone(model.trainer)
    
    def test_splitting_long_sequences(self):
        """Test fine-tuning with sequences longer than 512."""
        model = MethylBert(
            foundation_model_path=self.foundation_model,
            seq_len=600,  # Longer than 512
            custom_config=self.config,
            load_weights=False,
            num_labels=2,
            num_dmr_labels=10
        )
        
        from methyldl.data.dataset import generate_example_data_for_methylbert
        
        # generate_example_data_for_methylbert should handle splitting
        data = generate_example_data_for_methylbert(
            sequence_length=600,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=5
        )
        
        dataset = MethylBertFinetuneDataset(
            data_source=data,
            vocab=self.vocab,
            seq_len=600,
            n_cores=1
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
        from methyldl.data.dataset import generate_example_data_for_methylbert
        
        # Generate synthetic data
        data = generate_example_data_for_methylbert(
            sequence_length=self.seq_len,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=3
        )

        # Write to temp file (only dna_seq column for pretrain)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            # Skip header, write only dna sequences
            for row in data[1:]:
                f.write(row[0] + "\n")  # dna_seq column
            temp_file = f.name
        
        try:
            dataset = MethylBertPretrainDataset(
                f_path=temp_file,
                vocab=self.vocab,
                seq_len=self.seq_len
            )
            
            self.assertEqual(len(dataset), 3)
            
            # Test getitem
            item = dataset[0]
            self.assertIn("bert_input", item)
            self.assertIn("bert_label", item)
            self.assertIn("bert_mask", item)
            
        finally:
            os.unlink(temp_file)
    @parameterized.expand([
        (1),
        (5),
    ])
    def test_pretrain_dataset_masking(self,num_samples):
        """Test that masking is applied in pretrain dataset."""
        from methyldl.data.dataset import generate_example_data_for_methylbert
        
        # Generate a longer sequence to ensure masking
        data = generate_example_data_for_methylbert(
            sequence_length=self.seq_len,
            include_cpg_methylation=True,
            include_labels=True,
            num_samples=num_samples
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for row in data[1:]:
                f.write(row[0] + "\n")  # dna_seq column
            temp_file = f.name
        
        try:
            dataset = MethylBertPretrainDataset(
                f_path=temp_file,
                vocab=self.vocab,
                seq_len=self.seq_len
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
        from methyldl.data.dataset import generate_example_data_for_methylbert
        
        for seq_len in [50, 120, 200]:
            with self.subTest(seq_len=seq_len):
                data = generate_example_data_for_methylbert(
                    sequence_length=seq_len,
                    include_cpg_methylation=True,
                    include_labels=True,
                    num_samples=2
                )
                
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                    for row in data[1:]:
                        f.write(row[0] + "\n")
                    temp_file = f.name
                
                try:
                    dataset = MethylBertPretrainDataset(
                        f_path=temp_file,
                        vocab=self.vocab,
                        seq_len=seq_len
                    )
                    
                    self.assertGreater(len(dataset), 0)
                    
                    item = dataset[0]
                    # Check tensors have expected shape
                    self.assertEqual(item["bert_input"].shape[0], seq_len+1)
                    
                finally:
                    os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()