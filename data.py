import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class SummarizationDataset(Dataset):
    def __init__(self, tsv_file_path, transform=None):
        self.df = pd.read_csv(tsv_file_path, sep='\t')
        self.df.drop(self.df.columns.difference(['content', 'abstract']), 1, inplace=True)

        self.transform = transform

        if "content" not in self.df.columns or "abstract" not in self.df.columns:
            raise Exception('Either field "content" or "abstract" is missing !')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.iloc[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample["content"], sample["abstract"]


class SummarizationDataModule(pl.LightningDataModule):
    def __init__(self, train_file_path, validation_file_path, test_file_path, tokenizer, batch_size, num_workers,
                 src_max_seq_len, trg_max_seq_len):
        super().__init__()
        self.train_file_path = train_file_path
        self.validation_file_path = validation_file_path
        self.test_file_path = test_file_path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.src_max_seq_len = src_max_seq_len
        self.trg_max_seq_len = trg_max_seq_len

    def collate_fn(self, batch):
        batch_split = list(zip(*batch))

        sources, targets = batch_split[0], batch_split[1]
        sources = list(sources)
        targets = list(targets)

        encoded_contents = self.tokenizer(
            text=sources,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.src_max_seq_len,  # maximum length of a sentence
            padding=True,  # Add [PAD]s
            return_attention_mask=True,  # Generate the attention mask
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        encoded_abstracts = self.tokenizer(
            text=targets,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.trg_max_seq_len,  # maximum length of a sentence
            padding=True,  # Add [PAD]s
            return_attention_mask=True,  # Generate the attention mask
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )

        return encoded_contents, encoded_abstracts

    def train_dataloader(self):
        self.train_dataset = SummarizationDataset(self.train_file_path)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        self.validation_dataset = SummarizationDataset(self.validation_file_path)

        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        self.test_dataset = SummarizationDataset(self.test_file_path)

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
