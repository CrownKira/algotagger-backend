# Install a pip package in the current Jupyter kernel
# import sys
# !{sys.executable} -m pip install transformers
# !{sys.executable} -m pip install torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import cuda
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from ..preprocess.text import clean_questions


class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float),
        }


class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 10)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)  # type: ignore
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def validation(testing_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(
                device, dtype=torch.long
            )
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(
                torch.sigmoid(outputs).cpu().detach().numpy().tolist()
            )
    return fin_outputs, fin_targets


def predict_distilbert(title, body):
    question = title + " " + body
    testing = clean_questions([question])
    topics = [
        "array",
        "string",
        "dynamic_programming",
        "math",
        "hash_table",
        "greedy",
        "sorting",
        "depth_first_search",
        "breadth_first_search",
        "binary_search",
    ]
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", truncation=True, do_lower_case=True
    )
    model_path = "core/predictors/distilbert/pytorch_distilbert.bin"
    model = torch.load(model_path, map_location=torch.device("cpu"))
    device = "cuda" if cuda.is_available() else "cpu"
    test_data = pd.DataFrame()
    test_data["text"] = testing
    test_data["labels"] = [np.zeros(10)]
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4
    EPOCHS = 25
    LEARNING_RATE = 1e-05
    test_params = {
        "batch_size": VALID_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0,
    }
    testing_set = MultiLabelDataset(test_data, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set, **test_params)
    prediction, targets = validation(testing_loader, model, device)
    res = {}
    for i in range(len(topics)):
        res[topics[i]] = prediction[0][i]
    return res
