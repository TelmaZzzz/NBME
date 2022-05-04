import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import logging
from transformers import AutoModel, AutoConfig, T5EncoderModel, AutoModelForSeq2SeqLM
import numpy as np
import torch.nn.utils.rnn as rnn_utils
import Utils
from GlobalPointer import GlobalPointer


class ModelConfig(object):
    def __init__(self, args):
        self.pretrain_path = args.pretrain_path
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-7
        self.num_labels = args.num_labels
        self.device = args.device
        self.dropout = args.dropout
        self.rdrop = args.rdrop
        self.rdrop_alpha = args.rdrop_alpha


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):
    def __init__(self, name="swish"):
        super(Activation, self).__init__()
        if name not in ["swish", "relu", "gelu"]:
            raise
        if name == "swish":
            self.net = Swish()
        elif name == "relu":
            self.net = nn.ReLU()
        elif name == "gelu":
            self.net = nn.GELU()
    
    def forward(self, x):
        return self.net(x)


class Dence(nn.Module):
    def __init__(self, i_dim, o_dim, activation="swish"):
        super(Dence, self).__init__()
        self.dence = nn.Sequential(
            nn.Linear(i_dim, o_dim),
            # nn.ReLU(),
            Activation(activation),
        )

    def forward(self, x):
        return self.dence(x)


class NBMEModel(nn.Module):
    def __init__(self, args):
        super(NBMEModel, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_path)
        self.num_labels = args.num_labels
        if "t5" in args.pretrain_path:
            self.transformer = T5EncoderModel.from_pretrained(args.pretrain_path)
        else:
            self.transformer = AutoModel.from_pretrained(args.pretrain_path)
        # self.transformer.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(args.dropout)
        # self.GRU = nn.GRU(config.hidden_size, config.hidden_size // 2, batch_first=True, bidirectional=True)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        # self.output = nn.Sequential(
        #     Dence(config.hidden_size, config.hidden_size, "relu"),
        #     nn.Dropout(0.1),
        #     Dence(config.hidden_size, config.hidden_size // 4, "relu"),
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size // 4, self.num_labels),
        # )
        # self.output.apply(Utils.init_normal)
        self.output = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if token_type_ids is not None:
            transformer_out = self.transformer(input_ids, attention_mask, token_type_ids)
        else:
            transformer_out = self.transformer(input_ids, attention_mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        # sequence_output, _ = self.GRU(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        # logits = self.output(sequence_output)
        # logits_out = logits
        logits_out = torch.sigmoid(logits)
        loss = 0
        if labels is not None:
            # loss = self.loss(logits, labels)

            loss1 = self.loss(logits1, labels)
            loss2 = self.loss(logits2, labels)
            loss3 = self.loss(logits3, labels)
            loss4 = self.loss(logits4, labels)
            loss5 = self.loss(logits5, labels)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

        return logits_out, loss
    
    def loss(self, logits, labels):
        loss_fct = nn.BCEWithLogitsLoss()
        logits = logits.squeeze(-1)
        labels_mask = (labels != -1)
        # logging.debug(f"labels shape: {labels.shape}")
        # logging.debug(f"logits shape: {logits.shape}")
        # logging.debug(f"labels_mask shape: {labels_mask.shape}")
        active_logits = torch.masked_select(logits, labels_mask)
        true_labels = torch.masked_select(labels, labels_mask)
        loss = loss_fct(active_logits, true_labels)
        return loss


class NBMEGenModel(nn.Module):
    def __init__(self, args):
        super(NBMEGenModel, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_path)
        self.num_labels = args.num_labels
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(args.pretrain_path)
        # self.transformer.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(args.dropout)
        # self.GRU = nn.GRU(config.hidden_size, config.hidden_size // 2, batch_first=True, bidirectional=True)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.output = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, gen_labels=None):
        transformer_out = self.transformer(input_ids, attention_mask, labels=gen_labels)
        sequence_output = transformer_out.encoder_last_hidden_state 
        sequence_output = self.dropout(sequence_output)
        # sequence_output, _ = self.GRU(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        # logits = self.output(sequence_output)
        logits_out = torch.sigmoid(logits)
        loss = 0
        if labels is not None:
            # loss = self.loss(logits, labels)
            decoder_loss = transformer_out.loss
            loss1 = self.loss(logits1, labels)
            loss2 = self.loss(logits2, labels)
            loss3 = self.loss(logits3, labels)
            loss4 = self.loss(logits4, labels)
            loss5 = self.loss(logits5, labels)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            loss = 0.8 * loss + 0.2 * decoder_loss

        return logits_out, loss
    
    def loss(self, logits, labels):
        loss_fct = nn.BCEWithLogitsLoss()
        logits = logits.squeeze(-1)
        labels_mask = (labels != -1)
        # logging.debug(f"labels shape: {labels.shape}")
        # logging.debug(f"logits shape: {logits.shape}")
        # logging.debug(f"labels_mask shape: {labels_mask.shape}")
        active_logits = torch.masked_select(logits, labels_mask)
        true_labels = torch.masked_select(labels, labels_mask)
        loss = loss_fct(active_logits, true_labels)
        return loss



def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    #y_pred = (batch,l,l,c)
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_true = torch.reshape(y_true, (bh, -1))
    y_pred = torch.reshape(y_pred, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))


class NBMEGPModel(nn.Module):
    def __init__(self, args):
        super(NBMEGPModel, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_path)
        self.num_labels = args.num_labels
        if "t5" in args.pretrain_path:
            self.transformer = T5EncoderModel.from_pretrained(args.pretrain_path)
        else:
            self.transformer = AutoModel.from_pretrained(args.pretrain_path)
        # self.transformer.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(args.dropout)
        # self.GRU = nn.GRU(config.hidden_size, config.hidden_size // 2, batch_first=True, bidirectional=True)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.output = GlobalPointer(1, 64, config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if token_type_ids is not None:
            transformer_out = self.transformer(input_ids, attention_mask, token_type_ids)
        else:
            transformer_out = self.transformer(input_ids, attention_mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        # sequence_output, _ = self.GRU(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output), mask=attention_mask)
        logits2 = self.output(self.dropout2(sequence_output), mask=attention_mask)
        logits3 = self.output(self.dropout3(sequence_output), mask=attention_mask)
        logits4 = self.output(self.dropout4(sequence_output), mask=attention_mask)
        logits5 = self.output(self.dropout5(sequence_output), mask=attention_mask)
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        # logits = self.output(sequence_output, mask=attention_mask)
        loss = 0
        if labels is not None:
            # loss = self.loss(logits, labels)
            loss1 = self.loss(logits1, labels)
            loss2 = self.loss(logits2, labels)
            loss3 = self.loss(logits3, labels)
            loss4 = self.loss(logits4, labels)
            loss5 = self.loss(logits5, labels)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

        return logits, loss
    
    def loss(self, logits, labels):
        loss = global_pointer_crossentropy(labels, logits)
        return loss


def token_to_char(x, token_to_char_index):
    batch_size, L, dim = x.shape
    x = x.reshape(-1,dim)
    
    i = token_to_char_index + (torch.arange(batch_size)*L).reshape(-1,1).to(x.device)
    i = i.reshape(-1)
 
    c = x[i]
    c[i==0] = 0
    c = c.reshape(batch_size,-1,dim)
    return c


class NBMECharModel(nn.Module):
    def __init__(self, args):
        super(NBMECharModel, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_path)
        self.num_labels = args.num_labels
        self.rdrop = args.rdrop
        self.rdrop_alpha = args.rdrop_alpha
        if "t5" in args.pretrain_path:
            self.transformer = T5EncoderModel.from_pretrained(args.pretrain_path)
        else:
            self.transformer = AutoModel.from_pretrained(args.pretrain_path)
        # self.transformer.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(args.dropout)
        self.GRU = nn.GRU(config.hidden_size, config.hidden_size // 2, batch_first=True, bidirectional=True)
        self.init_gru(self.GRU)
        # self.GRU = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        # self.init_lstm(self.GRU)
        # self.output = nn.Sequential(
        #     Dence(config.hidden_size, config.hidden_size, "relu"),
        #     nn.Dropout(0.1),
        #     Dence(config.hidden_size, config.hidden_size // 4, "relu"),
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size // 4, self.num_labels),
        # )
        # self.output.apply(Utils.init_normal)
        self.token_output = nn.Linear(config.hidden_size, 1)
        self.char_output = nn.Linear(config.hidden_size, 1)

    def init_gru(self, cell, gain=1):
        cell.reset_parameters()
        # orthogonal initialization of recurrent weights
        for _, hh, _, _ in cell.all_weights:
            for i in range(0, hh.size(0), cell.hidden_size):
                I.orthogonal(hh[i:i + cell.hidden_size], gain=gain)


    def init_lstm(self, cell, gain=1):
        self.init_gru(cell, gain)
        # positive forget gate bias (Jozefowicz et al., 2015)
        for _, _, ih_b, hh_b in cell.all_weights:
            l = len(ih_b)
            ih_b[l // 4:l // 2].data.fill_(1.0)
            hh_b[l // 4:l // 2].data.fill_(1.0)

    def _forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, char_labels=None, token_to_char_index=None):
        if token_type_ids is not None:
            transformer_out = self.transformer(input_ids, attention_mask, token_type_ids=token_type_ids)
        else:
            transformer_out = self.transformer(input_ids, attention_mask)
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        char_loss = 0
        token_loss = 0
        
        if labels is not None:
            if self.training:
                token_loss = 0
                for dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    token_logits = self.token_output(F.dropout(sequence_output, dropout_rate, training=self.training))
                    token_loss += self.loss(token_logits, labels)
                token_loss = token_loss / 5
            else:
                token_logits = self.token_output(sequence_output)
                token_loss = self.loss(token_logits, labels)
        else:
            token_logits = self.token_output(sequence_output)
        
        char_sequence_output = token_to_char(sequence_output, token_to_char_index)
        char_sequence_output, _ = self.GRU(char_sequence_output)
        if char_labels is not None:
            if self.training:
                char_loss = 0
                for dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    # char_sequence_output = token_to_char(sequence_output, token_to_char_index)
                    # char_sequence_output, _ = self.GRU(F.dropout(char_sequence_output, dropout_rate, training=self.training))
                    char_logits = self.char_output(F.dropout(char_sequence_output, dropout_rate, training=self.training))
                    char_loss += self.loss(char_logits, char_labels)
                char_loss = char_loss / 5
            else:
                # char_sequence_output = token_to_char(sequence_output, token_to_char_index)
                # char_sequence_output, _ = self.GRU(char_sequence_output)
                char_logits = self.char_output(char_sequence_output)
                char_loss = self.loss(char_logits, char_labels)
        else:
            # char_sequence_output = token_to_char(sequence_output, token_to_char_index)
            # char_sequence_output, _ = self.GRU(char_sequence_output)
            char_logits = self.char_output(char_sequence_output)
        # loss = (char_loss + token_loss) / 2
        loss = char_loss
        # logits_out = char_logits
        logits_out = torch.sigmoid(char_logits)

        return logits_out, loss

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, char_labels=None, token_to_char_index=None):
        logits_out, loss = self._forward(input_ids, attention_mask, token_type_ids=token_type_ids, labels=labels, char_labels=char_labels, token_to_char_index=token_to_char_index)
        if self.training and self.rdrop:
            logits_out1, loss1 = self._forward(input_ids, attention_mask, token_type_ids=token_type_ids, labels=labels, char_labels=char_labels, token_to_char_index=token_to_char_index)
            ce_loss = (loss + loss1) / 2
            kl_loss = self.compute_kl_loss(logits_out, logits_out1, attention_mask)
            loss = ce_loss + self.rdrop_alpha * kl_loss
        return logits_out, loss

    def compute_kl_loss(self, p, q, pad_mask=None):
        pad_mask = torch.unsqueeze(pad_mask, -1).bool()
        p_loss = F.kl_div(p.log(), q, reduction='none', log_target=False)
        q_loss = F.kl_div(q.log(), p, reduction='none', log_target=False)
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss.mean()


    def loss(self, logits, labels):
        loss_fct = nn.BCEWithLogitsLoss()
        logits = logits.squeeze(-1)
        labels_mask = (labels != -1)
        # logging.debug(f"labels shape: {labels.shape}")
        # logging.debug(f"logits shape: {logits.shape}")
        # logging.debug(f"labels_mask shape: {labels_mask.shape}")
        active_logits = torch.masked_select(logits, labels_mask)
        true_labels = torch.masked_select(labels, labels_mask)
        loss = loss_fct(active_logits, true_labels)
        return loss
