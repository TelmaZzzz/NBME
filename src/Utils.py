import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import time
import logging
import os
import copy
import pandas as pd
from torch import Tensor
from typing import List, Optional
import ast
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import re


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def d2s(dt, time=False):
    if time is False:
        return dt.strftime("%Y_%m_%d")
    else:
        return dt.strftime("%Y_%m_%d_%H_%M_%S")


def timer(func):
    def deco(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logging.info("Function {} run {:.2f}s.".format(func.__name__, end_time - start_time))
        return res

    return deco


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.6, emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def init_normal(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.6)
        if m.bias is not None:
            m.bias.data.zero_()


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


@timer
def prepare_training_data(df, tokenizer, max_len):
    training_sample = []
    for _, row in df.iterrows():
        # if row["id"] == "10075_100":
        #     debug = True
        # else:
        #     debug = False
        pn_history = row["pn_history"]
        feature_text = row["feature_text"]
        encoder = tokenizer.encode_plus(
            pn_history, feature_text,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=False,
        )
        decoder = tokenizer.encode_plus(
            pn_history,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = decoder.offset_mapping
        ignore_index = np.where(np.array(decoder.sequence_ids()) != 0)[0]
        labels = np.zeros(len(offset_mapping))
        labels[ignore_index] = -1
        location_list = eval(row["location"])
        for location in location_list:
            for loc in [s.split() for s in location.split(";")]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx - 1
                if end_idx > start_idx:
                    labels[start_idx: end_idx] = 1
        training_sample.append({
            "input_ids": encoder.input_ids,
            "attention_mask": encoder.attention_mask,
            # "token_type_ids": encoder.token_type_ids,
            "offset_mapping": offset_mapping,
            "labels": labels,
            "id": row["id"],
            "length": len(pn_history),
        })
    return training_sample


@timer
def prepare_training_data_fix(df, tokenizer, max_len):
    training_sample = []
    for _, row in df.iterrows():
        # if row["id"] == "10075_100":
        #     debug = True
        # else:
        #     debug = False
        pn_history = row["pn_history"]
        feature_text = row["feature_text"]
        encoder = tokenizer.encode_plus(
            pn_history, feature_text,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=False,
        )
        decoder = tokenizer.encode_plus(
            pn_history,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = decoder.offset_mapping
        for i in range(1, len(offset_mapping)):
            if offset_mapping[i][0] < offset_mapping[i][1]:
                if offset_mapping[i][0] != offset_mapping[i-1][1]:
                    offset_mapping[i] = (offset_mapping[i][0] - 1, offset_mapping[i][1])
            
        ignore_index = np.where(np.array(decoder.sequence_ids()) != 0)[0]
        labels = np.zeros(len(offset_mapping))
        labels[ignore_index] = -1
        location_list = eval(row["location"])
        for location in location_list:
            for loc in [s.split() for s in location.split(";")]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx - 1
                if end_idx > start_idx:
                    labels[start_idx: end_idx] = 1
        training_sample.append({
            "input_ids": encoder.input_ids,
            "attention_mask": encoder.attention_mask,
            # "token_type_ids": encoder.token_type_ids,
            "offset_mapping": offset_mapping,
            "labels": labels,
            "id": row["id"],
            "length": len(pn_history),
        })
    return training_sample


@timer
def prepare_training_data_gen(df, tokenizer, max_len):
    training_sample = []
    for _, row in df.iterrows():
        # if row["id"] == "10075_100":
        #     debug = True
        # else:
        #     debug = False
        pn_history = row["pn_history"]
        feature_text = row["feature_text"]
        encoder = tokenizer.encode_plus(
            pn_history, feature_text,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=False,
        )
        decoder = tokenizer.encode_plus(
            pn_history,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        annotation = eval(row["annotation"])
        gen_labels = " # ".join(annotation)
        gen = tokenizer.encode_plus(
            gen_labels,
            max_length=128,
            padding="max_length",
            add_special_tokens=True,
        ).input_ids
        gen = [-100 if item == tokenizer.pad_token_id else item for item in gen]
        offset_mapping = decoder.offset_mapping
        ignore_index = np.where(np.array(decoder.sequence_ids()) != 0)[0]
        labels = np.zeros(len(offset_mapping))
        labels[ignore_index] = -1
        location_list = eval(row["location"])
        for location in location_list:
            for loc in [s.split() for s in location.split(";")]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx - 1
                if end_idx > start_idx:
                    labels[start_idx: end_idx] = 1
        training_sample.append({
            "input_ids": encoder.input_ids,
            "attention_mask": encoder.attention_mask,
            # "token_type_ids": encoder.token_type_ids,
            "offset_mapping": offset_mapping,
            "labels": labels,
            "id": row["id"],
            "length": len(pn_history),
            "gen_labels": gen,
        })
    return training_sample


def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)


def create_labels_for_scoring(df):
    df["location_for_create_labels"] = df["location"]
    truths = []
    for location_list in df['location_for_create_labels'].values:
        location_list = eval(location_list)
        truth = []
        for location in location_list:
            for loc in [s.split(" ") for s in location.split(";")]:
                truth.append([int(loc[0]), int(loc[1])])
        truths.append(truth)
    return truths


def get_char_probs(samples, predictions):
    results = [np.zeros(item["length"]) for item in samples]
    for i, (sample, prediction) in enumerate(zip(samples, predictions)):
        offset_mapping = sample["offset_mapping"]
        for idx, (offset_mapping, pred) in enumerate(zip(offset_mapping, prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results


def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results


def get_results_v2(char_probs, valid_texts, th=0.5):
    results = []
    for i, char_prob in enumerate(char_probs):
        result = []
        end = 0
        while end < len(char_prob):
            if char_prob[end] < th:
                end += 1
            else:
                start = end
                while end < len(char_prob) and char_prob[end] >= th:
                    end += 1
                if valid_texts[i][start].isspace():
                    result.extend(list(range(start+1, end+1)))
                else:
                    result.extend(list(range(start, end+1)))

        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results


def get_predictions(results):
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)
    return predictions


def text_to_word(text):
    word = text.split()
    word_offset = []

    start = 0
    pre = 0
    for w in word:
        r = text[start:].find(w)

        if r == -1:
            raise NotImplementedError
        else:
            start = start + r
            end = start + len(w)
            word_offset.append((pre, end))
            pre = end
            # print('%32s'%w, '%5d'%start, '%5d'%r, text[start:end])
        start = end

    return word, word_offset


def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        device=None,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.device = device
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            _, adv_loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            self.optimizer.zero_grad()
            adv_loss.backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class PGD():
    def __init__(self, model, emb_name="embeddings.", epsilon=0.6, alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


@timer
def prepare_valid_data(df, tokenizer, max_len):
    training_sample = []
    for _, row in df.iterrows():
        # if row["id"] == "10075_100":
        #     debug = True
        # else:
        #     debug = False
        pn_history = row["pn_history"]
        feature_text = row["feature_text"]
        encoder = tokenizer.encode_plus(
            pn_history, feature_text,
            add_special_tokens=True,
            return_offsets_mapping=False,
        )
        decoder = tokenizer.encode_plus(
            pn_history,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = decoder.offset_mapping
        ignore_index = np.where(np.array(decoder.sequence_ids()) != 0)[0]
        labels = np.zeros(len(offset_mapping))
        labels[ignore_index] = -1
        location_list = eval(row["location"])
        for location in location_list:
            for loc in [s.split() for s in location.split(";")]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx - 1
                if end_idx > start_idx:
                    labels[start_idx: end_idx] = 1
        training_sample.append({
            "input_ids": encoder.input_ids,
            "attention_mask": encoder.attention_mask,
            "offset_mapping": offset_mapping,
            "labels": labels,
            "id": row["id"],
            "length": len(pn_history),
        })
    return training_sample


def get_char_probs_gp(samples, predictions):
    results = [np.array([0]*item["length"]) for item in samples]
    for idx, (sample, prediction) in enumerate(zip(samples, predictions)):
        offset_mapping = sample["offset_mapping"]
        p = np.zeros(len(offset_mapping))
        tmp = np.where(prediction > 0)
        for i, j in zip(tmp[1], tmp[2]):
            p[i:j+1]=1
        tmp = np.where(p == 1)[0]
        for i in tmp:
            start = offset_mapping[i][0]
            end = offset_mapping[i][1]
            results[idx][start:end]=1
    return results


def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.greater(y_pred, 0)
    # pre_index = y_pred.nonzero() #获取实体的索引[batch_index,type_index,start_index,end_index]
    # l = y_true * y_pred #预测正确的数量
    # h = y_true + y_pred #预测的数量+真实的数量
    return torch.sum(y_true * y_pred).item(), torch.sum(y_true + y_pred).item()


def get_char_probs_char(samples, predictions):
    results = [np.zeros(item["length"]) for item in samples]
    for i, (sample, prediction) in enumerate(zip(samples, predictions)):
        length = sample["length"]
        results[i][:length] = prediction[:length]
    return results


@timer
def prepare_training_data_char(df, tokenizer, max_len):
    training_sample = []
    for _, row in df.iterrows():
        # if row["id"] == "10075_100":
        #     debug = True
        # else:
        #     debug = False
        pn_history = row["pn_history"]
        feature_text = row["feature_text"]
        encoder = tokenizer.encode_plus(
            pn_history, feature_text,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=False,
        )
        decoder = tokenizer.encode_plus(
            pn_history,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = decoder.offset_mapping
        ignore_index = np.where(np.array(decoder.sequence_ids()) != 0)[0]
        labels = np.zeros(len(offset_mapping))
        labels[ignore_index] = -1
        location_list = eval(row["location"])
        char_labels = np.zeros(1024)
        for location in location_list:
            for loc in [s.split() for s in location.split(";")]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                char_labels[start: end] = 1
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx - 1
                if end_idx > start_idx:
                    labels[start_idx: end_idx] = 1
        char_labels[len(pn_history):] = -1
        training_sample.append({
            "input_ids": encoder.input_ids,
            "attention_mask": encoder.attention_mask,
            # "token_type_ids": encoder.token_type_ids,
            "offset_mapping": offset_mapping,
            "labels": labels,
            "id": row["id"],
            "length": len(pn_history),
            "char_labels": char_labels,
        })
    return training_sample


@timer
def prepare_training_data_word(df, tokenizer, max_len):
    training_sample = []
    for _, row in df.iterrows():
        # if row["id"] == "10075_100":
        #     debug = True
        # else:
        #     debug = False
        pn_history = row["pn_history"]
        feature_text = row["feature_text"]
        encoder = tokenizer.encode_plus(
            pn_history, feature_text,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=False,
        )
        decoder = tokenizer.encode_plus(
            pn_history,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = decoder.offset_mapping
        ignore_index = np.where(np.array(decoder.sequence_ids()) != 0)[0]
        labels = np.zeros(len(offset_mapping))
        labels[ignore_index] = -1
        location_list = eval(row["location"])
        word_labels = np.zeros(512)
        word_list, word_offset_mapping = text_to_word(pn_history)
        for location in location_list:
            for loc in [s.split() for s in location.split(";")]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx - 1
                if end_idx > start_idx:
                    labels[start_idx: end_idx] = 1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(word_offset_mapping)):
                    if (start_idx == -1) & (start < word_offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= word_offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx - 1
                if end_idx > start_idx:
                    word_labels[start_idx: end_idx] = 1
        word_offset_mapping += [(0, 0)] * (512 - len(word_offset_mapping))
        word_labels[len(word_list):] = -1
        training_sample.append({
            "input_ids": encoder.input_ids,
            "attention_mask": encoder.attention_mask,
            # "token_type_ids": encoder.token_type_ids,
            "offset_mapping": offset_mapping,
            "labels": labels,
            "id": row["id"],
            "length": len(pn_history),
            "word_labels": word_labels,
            "word_offset_mapping": word_offset_mapping,
        })
    return training_sample


@timer
def prepare_test_data(df, tokenizer, max_len):
    test_sample = []
    for _, row in df.iterrows():
        # if row["id"] == "10075_100":
        #     debug = True
        # else:
        #     debug = False
        pn_history = row["pn_history"]
        feature_text = row["feature_text"]
        encoder = tokenizer.encode_plus(
            pn_history, feature_text,
            add_special_tokens=True,
            return_offsets_mapping=False,
        )
        decoder = tokenizer.encode_plus(
            pn_history,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = decoder.offset_mapping
        test_sample.append({
            "input_ids": encoder.input_ids,
            "attention_mask": encoder.attention_mask,
            "offset_mapping": offset_mapping,
            "id": row["id"],
            "length": len(pn_history),
        })
    return test_sample


@timer
def prepare_training_data_da(df, tokenizer, max_len):
    training_sample = []
    for _, row in df.iterrows():
        # if row["id"] == "10075_100":
        #     debug = True
        # else:
        #     debug = False
        pn_history = row["pn_history"]
        feature_text = row["feature_text"]
        encoder = tokenizer.encode_plus(
            pn_history, feature_text,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=False,
        )
        decoder = tokenizer.encode_plus(
            pn_history,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = decoder.offset_mapping
        ignore_index = np.where(np.array(decoder.sequence_ids()) != 0)[0]
        labels = np.zeros(len(offset_mapping))
        labels[ignore_index] = -1
        location_list = row["location"]
        # logging.info(location_list)
        for loc in [s.split() for s in location_list.split(";")]:
            start_idx = -1
            end_idx = -1
            if len(loc)==0:
                continue
            start, end = int(loc[0]), int(loc[1])
            # logging.info(f"rep: {start} {end}")
            for idx in range(len(offset_mapping)):
                if (start_idx == -1) & (start < offset_mapping[idx][0]):
                    start_idx = idx - 1
                if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                    end_idx = idx + 1
            if start_idx == -1:
                start_idx = end_idx - 1
            if end_idx > start_idx:
                labels[start_idx: end_idx] = 1
        training_sample.append({
            "input_ids": encoder.input_ids,
            "attention_mask": encoder.attention_mask,
            # "token_type_ids": encoder.token_type_ids,
            "offset_mapping": offset_mapping,
            "labels": labels,
            "id": row["id"],
            "length": len(pn_history),
        })
    return training_sample


def get_results_v3(char_probs, valid_texts, valid_id, th_mp={}, th=0.5):
    results = []
    for i, char_prob in enumerate(char_probs):
        result = []
        end = 0
        feature_num = valid_id[i].split("_")[-1]
        fth = th
        ### post 3
        fth = th_mp.get(feature_num, 0.5)
        ### /post 3
        ### post 2
        # if feature_num == "207":
        #     fth = 0.35
        # if feature_num[0] == "7":
        #     fth = 0.55
        # if feature_num == "313":
        #     fth = 0.50
        # if feature_num == "107":
        #     fth = 0.45
        ### /post 2
        while end < len(char_prob):
            if char_prob[end] < fth:
                end += 1
            else:
                start = end
                while end < len(char_prob) and char_prob[end] >= fth:
                    end += 1
                ### post 1
                if feature_num == "708":
                    if "5" in valid_texts[i][start:end+1] or "3" in valid_texts[i][start:end+1]:
                        # logging.info(valid_texts[i][start:end+1])
                        continue
                ### /post 1
                if valid_texts[i][start].isspace():
                    result.extend(list(range(start+1, end+1)))
                else:
                    result.extend(list(range(start, end+1)))

        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results


def get_score_all(y_true, y_pred):
    return span_micro_f1_all(y_true, y_pred)


def span_micro_f1_all(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1_all(bin_preds, bin_truths)


def micro_f1_all(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds), precision_score(truths, preds), recall_score(truths, preds)


def get_results_v2_year(char_probs, valid_texts, th=0.5):
    results = []
    for i, char_prob in enumerate(char_probs):
        begin_th = th
        new_th = th

        pred_cnt = (char_prob > th).sum()
        if pred_cnt / len(valid_texts[i]) < 0.025:
            begin_th = th
            new_th = th * 0.45
        elif pred_cnt / len(valid_texts[i]) < 0.05:
            begin_th = th
            new_th = th * 0.7
        elif pred_cnt / len(valid_texts[i]) > 0.15:
            begin_th = begin_th * 1.25
            new_th = th * 1.1
        elif pred_cnt / len(valid_texts[i]) > 0.1:
            begin_th = begin_th * 1.2
            new_th = th * 1.2
        
        result = []
        end = 0
        while end < len(char_prob):
            if char_prob[end] < begin_th:
                end += 1
            else:
                start = end
                while end < len(char_prob) and char_prob[end] >= new_th:
                    end += 1
                
                if valid_texts[i][start].isspace():
                    start += 1

                matches = list(re.finditer(r'\d+\s*', valid_texts[i][start: end]))
                if len(matches) > 0 and matches[-1].span()[1] == end - start:
                    match = re.match(
                        r'\s*((year(s){0, 1}-old)|(year(s){0, 1} old)|(yr(s){0, 1})|(yo)|(year(s)*(\s*ago)*)|(y-o))',
                        valid_texts[i][end:],
                        re.I | re.M,
                    )
                    if match is not None:
                        end += match.span()[1]
                result.extend(list(range(start, end+1)))
        
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]

        result = ";".join(result)
        results.append(result)
    return results
