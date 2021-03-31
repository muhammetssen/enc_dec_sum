import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class SummarizationDataset(Dataset):
    def __init__(self, tsv_file_path, source_field_name="content",
                 target_field_name="abstract", transform=None):
        self.df = pd.read_csv(tsv_file_path, sep='\t')
        self.df.drop(self.df.columns.difference([source_field_name, target_field_name]), 1, inplace=True)

        self.transform = transform

        if source_field_name not in self.df.columns or target_field_name not in self.df.columns:
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
    def __init__(self, train_file_path, validation_file_path, test_file_path, tokenizer, model_name, batch_size,
                 num_workers, src_max_seq_len=512, trg_max_seq_len=128, dataset_name=None, source_field_name="content",
                 target_field_name="abstract"):
        super().__init__()
        self.train_file_path = train_file_path
        self.validation_file_path = validation_file_path
        self.test_file_path = test_file_path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.src_max_seq_len = src_max_seq_len
        self.trg_max_seq_len = trg_max_seq_len
        self.dataset_name = dataset_name
        self.source_field_name = source_field_name
        self.target_field_name = target_field_name
        self.model_name = model_name

        if any(x not in self.model_name for x in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]):
            self.prefix = "summarize: "
        else:
            self.prefix = ""

    def collate_fn(self, batch):
        batch_split = list(zip(*batch))

        sources, targets = batch_split[0], batch_split[1]
        sources = list(sources)
        targets = list(targets)

        sources = [self.prefix + doc for doc in sources]

        encoded_contents = self.tokenizer(
            text=sources,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.src_max_seq_len,  # maximum length of a sentence
            padding=True,  # Add [PAD]s
            truncation=True,
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        with self.tokenizer.as_target_tokenizer():
            encoded_abstracts = self.tokenizer(
                text=targets,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length=self.trg_max_seq_len,  # maximum length of a sentence
                padding=True,  # Add [PAD]s
                truncation=True,
                return_tensors="pt",  # ask the function to return PyTorch tensors
            )

        source_ids = encoded_contents['input_ids'].squeeze()
        source_mask = encoded_contents['attention_mask'].squeeze()
        target_ids = encoded_abstracts['input_ids'].squeeze()
        target_mask = encoded_abstracts['attention_mask'].squeeze()
        # labels = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels] for labels in target_ids]

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

    def train_dataloader(self):
        train_dataset = SummarizationDataset(self.train_file_path, self.source_field_name, self.target_field_name)

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        validation_dataset = SummarizationDataset(self.validation_file_path, self.source_field_name,
                                                  self.target_field_name)

        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        test_dataset = SummarizationDataset(self.test_file_path, self.source_field_name, self.target_field_name)

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
