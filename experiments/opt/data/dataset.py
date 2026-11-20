"""Module for datasets classes"""
import os

import pandas as pd
import torch
from torch import tensor
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

model_file = "FacebookAI/roberta-base"
# Dataset classes
class TweetSentimentDataset(Dataset):
    """Dataset Class for TwitterSentiment140 dataset

    params
    ---------------
    + csv_path : file path to the dataset

    + freq_threshold : Used during constructing vocabulary. Defaults to 5

    + vocab_file : If given vocabulary will be loaded from this file

    """

    def __init__(self, csv_path, num_words=100,  dataidxs=None):
        self.df = pd.read_csv(csv_path)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df['target'].replace({4: 1}, inplace=True)

        self.tokenizer = RobertaTokenizer.from_pretrained(model_file)
        self.num_words = num_words
        self.data = self.df['cleaned_text']
        self.target = self.df['target']
        self.dataidxs = dataidxs

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = self.target[self.dataidxs]
            self.data = self.data.reset_index(drop=True)
            self.target = self.target.reset_index(drop=True)


    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        tweet = self.data[idx]
        target = self.target[idx]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.num_words,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.long)
        }

