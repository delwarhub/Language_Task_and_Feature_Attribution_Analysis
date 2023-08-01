import gc
import pandas as pd
import itertools
from typing import Literal
from tqdm import tqdm
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

_data_types = Literal["train", "validation", "test"]

class classification_dataset:
    def __init__(self, config: dict):
        super(classification_dataset, self).__init__()

        self.config = config
        self.model_checkpoint = self.config["MODEL_CHECKPOINT"]
        self.source_column = self.config["SOURCE_COLUMN"]
        self.target_column = self.config["TARGET_COLUMN"]
        self.max_seq_len = self.config["MAX_SEQUENCE_LEN"]
        self.batch_size = self.config["BATCH_SIZE"]

        self.data = load_dataset(self.config["DATASET_LABEL"])

        self.class_labels = ['pop_culture', 'daily_life', 'sports_&_gaming', 'arts_&_culture', 'business_&_entrepreneurs', 'science_&_technology']

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def retrieve_dataframe(self, data_type: _data_types):
        data_type = f"{data_type}_2021"
        dataset = pd.DataFrame(self.data[data_type])
        dataset.drop(columns=["date", "id", "label_name"], axis=1, inplace=True)
        dataset = dataset[[self.source_column, self.target_column]]
        return dataset[:5]

    def process_data(self, data_type: _data_types):
        dataset = self.retrieve_dataframe(data_type)

        # tokenize input-texts
        source = [s for s in dataset[self.source_column].values.tolist()]
        model_inputs = self.tokenizer(source,
                                      max_length=self.max_seq_len,
                                      padding="max_length",
                                      truncation=True)

        labels = dataset[self.target_column].tolist()
        model_inputs["labels"] = labels
        model_inputs["input_ids"] = torch.tensor([i for i in model_inputs["input_ids"]], dtype=torch.long, device=self.device)
        model_inputs["attention_mask"] = torch.tensor([i for i in model_inputs["attention_mask"]], dtype=torch.long, device=self.device)
        model_inputs["labels"] = torch.tensor([i for i in model_inputs["labels"]], dtype=torch.long, device=self.device)

        del dataset
        del source
        gc.collect()
        return model_inputs

    def set_up_data_loader(self, data_type: _data_types):
        dataset = self.process_data(data_type)
        dataset = TensorDataset(dataset["input_ids"],
                                dataset["attention_mask"],
                                dataset["labels"])
        gc.collect()
        return DataLoader(dataset,
                          batch_size=self.batch_size)
