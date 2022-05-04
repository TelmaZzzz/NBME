import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import logging
import Utils
import random


class NBMEDataset:
    def __init__(self, samples, tokenizer, mask_prob=0.0, mask_ratio=0.0):
        self.samples = samples
        self.length = len(samples)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        attention_mask = self.samples[idx]["attention_mask"]
        labels = self.samples[idx]["labels"]
        # token_type_ids = self.samples[idx]["token_type_ids"]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # mask argument
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


class NBMEGenDataset:
    def __init__(self, samples, tokenizer, mask_prob=0.0, mask_ratio=0.0):
        self.samples = samples
        self.length = len(samples)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        attention_mask = self.samples[idx]["attention_mask"]
        labels = self.samples[idx]["labels"]
        gen_labels = self.samples[idx]["gen_labels"]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # mask argument
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
            "gen_labels": torch.tensor(gen_labels, dtype=torch.long),
        }



class NBMEDatasetValid:
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.length = len(samples)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        attention_mask = self.samples[idx]["attention_mask"]
        # labels = self.samples[idx]["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "labels": labels,
        }


class Collate:
    def __init__(self, tokenizer, fix_length=-1, fixed=False):
        self.tokenizer = tokenizer
        self.fix_length = fix_length
        self.fixed = fixed

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        # output["labels"] = [sample["labels"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])
        if self.fix_length != -1:
            batch_max = min(batch_max, self.fix_length)
            if self.fixed:
                batch_max = self.fix_length

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
            # output["labels"] = [s + (batch_max - len(s)) * [-1] for s in output["labels"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]
            # output["labels"] = [(batch_max - len(s)) * [-1] + s for s in output["labels"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        # output["labels"] = torch.tensor(output["labels"], dtype=torch.float)

        return output


class NBMEGPDataset:
    def __init__(self, samples, tokenizer, mask_prob=0.0, mask_ratio=0.0):
        self.samples = samples
        self.length = len(samples)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        attention_mask = self.samples[idx]["attention_mask"]
        labels = self.samples[idx]["labels"]
        gp_labels = torch.zeros((1, len(labels), len(labels)))
        end = 0
        LEN = len(labels)
        while end < LEN:
            if labels[end] != 1:
                end += 1
                continue
            start = end
            while end < LEN:
                if labels[end] == 1:
                    end += 1
                else:
                    break
            gp_labels[0,start,end-1] = 1
        # token_type_ids = self.samples[idx]["token_type_ids"]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # mask argument
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
            "gp_labels": gp_labels
        }


class NBMECharDataset:
    def __init__(self, samples, tokenizer, mask_prob=0.0, mask_ratio=0.0):
        self.samples = samples
        self.length = len(samples)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        attention_mask = self.samples[idx]["attention_mask"]
        labels = self.samples[idx]["labels"]
        offset_mapping = self.samples[idx]["offset_mapping"]
        char_labels = self.samples[idx]["char_labels"]
        # token_type_ids = self.samples[idx]["token_type_ids"]
        token_to_char_index = np.zeros(1024)
        for i, (label, offset) in enumerate(zip(labels, offset_mapping)):
            start = offset[0]
            end = offset[1]
            token_to_char_index[start:end] = i

        # token_type_ids = self.samples[idx]["token_type_ids"]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # mask argument
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
            "char_labels": torch.tensor(char_labels, dtype=torch.float),
            "token_to_char_index": torch.tensor(token_to_char_index, dtype=torch.long),
        }


class NBMECharDatasetValid:
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.length = len(samples)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        attention_mask = self.samples[idx]["attention_mask"]
        offset_mapping = self.samples[idx]["offset_mapping"]
        token_to_char_index = np.zeros(1024)
        for i, offset in enumerate(offset_mapping):
            start = offset[0]
            end = offset[1]
            token_to_char_index[start:end] = i

        # token_type_ids = self.samples[idx]["token_type_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "token_to_char_index": token_to_char_index,
        }


class CollateChar:
    def __init__(self, tokenizer, fix_length=-1, fixed=False):
        self.tokenizer = tokenizer
        self.fix_length = fix_length
        self.fixed = fixed

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["token_to_char_index"] = [sample["token_to_char_index"] for sample in batch]
        # output["labels"] = [sample["labels"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])
        if self.fix_length != -1:
            batch_max = min(batch_max, self.fix_length)
            if self.fixed:
                batch_max = self.fix_length

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
            # output["labels"] = [s + (batch_max - len(s)) * [-1] for s in output["labels"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]
            # output["labels"] = [(batch_max - len(s)) * [-1] + s for s in output["labels"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        output["token_to_char_index"] = torch.tensor(output["token_to_char_index"], dtype=torch.long)
        # output["labels"] = torch.tensor(output["labels"], dtype=torch.float)

        return output


class NBMEWordDataset:
    def __init__(self, samples, tokenizer, mask_prob=0.0, mask_ratio=0.0):
        self.samples = samples
        self.length = len(samples)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        attention_mask = self.samples[idx]["attention_mask"]
        labels = self.samples[idx]["labels"]
        offset_mapping = self.samples[idx]["offset_mapping"]
        char_labels = self.samples[idx]["word_labels"]
        word_offset_mapping = self.samples[idx]["word_offset_mapping"]
        # token_type_ids = self.samples[idx]["token_type_ids"]
        token_to_char_index = np.zeros(1024)
        for i, (label, offset) in enumerate(zip(labels, offset_mapping)):
            start = offset[0]
            end = offset[1]
            token_to_char_index[start:end] = i

        # token_type_ids = self.samples[idx]["token_type_ids"]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # mask argument
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
            "char_labels": torch.tensor(char_labels, dtype=torch.float),
            "token_to_char_index": torch.tensor(token_to_char_index, dtype=torch.long),
        }