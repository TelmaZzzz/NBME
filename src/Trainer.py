import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import Model, Utils, Datasets
from transformers import AdamW, get_cosine_schedule_with_warmup
import logging
import gc
import torch.cuda.amp as AMP
from apex import amp
from tqdm import tqdm
import numpy as np


class TrainerConfig(object):
    def __init__(self, args):
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epoch = args.epoch
        self.opt_step = args.opt_step
        self.eval_step = args.eval_step
        self.Tmax = args.Tmax
        self.min_lr = args.min_lr
        self.scheduler = args.scheduler
        self.max_norm = args.max_norm
        self.model_save = args.model_save
        self.model_load = args.model_load
        self.metrics = args.metrics
        self.model_name = args.model_name
        self.debug = args.debug
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.fp16 = args.fp16
        self.fgm = args.fgm
        self.radam = args.radam
        self.freeze_step = args.freeze_step
        self.fix_length = args.fix_length
        self.ema = args.ema
        self.awp = args.awp
        self.adv_lr = args.adv_lr
        self.adv_eps = args.adv_eps
        self.awp_up = args.awp_up
        self.fgm_up = args.fgm_up
        self.pgd = args.pgd
        self.pgd_k = args.pgd_k
        self.swa = args.swa
        self.swa_start_step = args.swa_start_step
        self.swa_update_step = args.swa_update_step
        self.swa_lr = args.swa_lr
        self.warmup_step = args.warmup_step


class BaseTrainer(object):
    def __init__(self, args):
        self.predict_loss = 0
        self.trainer_config = TrainerConfig(args)
        self.model_config = Model.ModelConfig(args)
        self.device = args.device

    def build_model(self):
        self.model = Model.NBMEModel(self.model_config)

    def model_init(self):
        self.build_model()
        if self.trainer_config.model_load:
            self.model.load_state_dict(torch.load(self.trainer_config.model_load, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.train()

    def optimizer_init(self):
        optimizer_grouped_parameters = self._get_optimizer_grouped_parameters()
        self.optimizer = AdamW(optimizer_grouped_parameters)
        training_epoch = self.training_size // self.trainer_config.epoch
        scheduler_map = {
            "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.trainer_config.Tmax, eta_min=self.trainer_config.min_lr),
            "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.trainer_config.Tmax, T_mult=1, eta_min=self.trainer_config.min_lr),
            "get_cosine_schedule_with_warmup": get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=int(0.1 * self.training_size) if self.trainer_config.warmup_step == -1 else self.trainer_config.warmup_step, 
                num_training_steps=self.training_size,
                num_cycles=1,
                last_epoch=-1,
            ),
            "MultiStepLR": lr_scheduler.MultiStepLR(self.optimizer, [training_epoch * 2, training_epoch * 6], gamma=0.1),
        }
        if self.trainer_config.fp16:
            self.model, self.optmizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        if self.trainer_config.fgm:
            self.fgm = Utils.FGM(self.model)
        if self.trainer_config.ema:
            self.ema = Utils.EMA(self.model, 0.999)
            self.ema.register()
        if self.trainer_config.awp:
            self.awp = Utils.AWP(self.model, self.optimizer, adv_lr=self.trainer_config.adv_lr, device=self.device, \
                adv_eps=self.trainer_config.adv_eps, start_epoch=self.training_size // self.trainer_config.epoch)
        if self.trainer_config.pgd:
            self.pgd = Utils.PGD(self.model)
        # if self.trainer_config.swa:
        #     self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        #     self.swa_model.eval()
        #     self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, swa_lr=self.trainer_config.swa_lr)
        #     self.swa_flag = False
        self.scheduler = scheduler_map[self.trainer_config.scheduler]
        self.f1_maxn = 0
        self.num_step = 0

    def _get_optimizer_grouped_parameters(self):
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n in no_decay
                ],
                "weight_decay": 0,
                "lr": self.trainer_config.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in no_decay
                ],
                "weight_decay": self.trainer_config.weight_decay,
                "lr": self.trainer_config.lr,
            },
        ]
        return optimizer_grouped_parameters

    def get_logits(self, batch, return_loss=False, use_swa=False):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        # token_type_ids = batch["token_type_ids"].to(self.device)
        # logging.debug(f"token_type_ids: {token_type_ids.shape}")
        if return_loss:
            labels = batch["labels"].to(self.device)
            # logging.debug(f"labels: {labels.tolist()}")
            # logging.debug(f"ids: {input_ids.tolist()}")
            logits, loss = self.model(input_ids, attention_mask, labels=labels)
            # logits, loss = self.model(input_ids, attention_mask, labels=labels)
            return logits, loss
        else:
            # if use_swa:
            #     logits, _ = self.swa_model(input_ids, attention_mask)
            # else:
            logits, _ = self.model(input_ids, attention_mask)
            # logits, _ = self.model(input_ids, attention_mask)
            return logits
    
    def get_loss(self, batch):
        _, loss = self.get_logits(batch, return_loss=True)
        return loss
    
    def step(self, batch):
        if self.trainer_config.freeze_step != -1 and self.num_step % self.trainer_config.freeze_step == 0:
            self.freeze((self.num_step // self.trainer_config.freeze_step) % 2)
        loss = self.get_loss(batch)
        loss /= self.trainer_config.opt_step
        self.num_step += 1
        if self.trainer_config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if self.trainer_config.fgm and self.f1_maxn >= self.trainer_config.fgm_up:
            self.fgm.attack()
            loss_fgm = self.get_loss(batch)
            if self.trainer_config.fp16:
                with amp.scale_loss(loss_fgm, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_fgm.backward()
            self.fgm.restore()
        if self.trainer_config.pgd:
            self.pgd.backup_grad()
            for t in range(self.trainer_config.pgd_k):
                self.pgd.attack(is_first_attack=(t==0))
                if t != self.trainer_config.pgd_k-1:
                    self.model.zero_grad()
                else:
                    self.pgd.restore_grad()
                with AMP.autocast(enabled=self.trainer_config.fp16):
                    loss_pgd = self.get_loss(batch)
                if self.trainer_config.fp16:
                    with amp.scale_loss(loss_pgd, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_pgd.backward()
            self.pgd.restore()
        if self.trainer_config.awp and self.f1_maxn > self.trainer_config.awp_up:
            self.awp.attack_backward(batch, self.num_step)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.max_norm)
        if self.num_step % self.trainer_config.opt_step == 0:
            self.optimizer.step()
            if self.trainer_config.ema:
                self.ema.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if self.num_step % self.trainer_config.eval_step == 0:
                self.eval()
        # if self.trainer_config.swa and self.num_step >= self.trainer_config.swa_start_step:
        #     self.swa_flag = True
        #     self.scheduler = self.swa_scheduler
        # if self.trainer_config.swa and self.swa_flag and self.num_step % self.trainer_config.swa_update_step == 0:
        #     self.swa_model.update_parameters(self.model)
        return loss.cpu()

    @torch.no_grad()
    def eval(self, valid_datasets=None, valid_collate=None):
        if self.trainer_config.ema:
            self.ema.apply_shadow()
        # if self.trainer_config.swa and self.swa_flag:
        #     self.swa_model.eval()
        # else:
        self.model.eval()
        if valid_datasets is None:
            valid_datasets = self.valid_datasets
        if valid_collate is None:
            valid_collate = self.valid_collate
        if valid_collate is None:
            valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size)
        else:
            valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size, collate_fn=valid_collate)
        preds = []
        # labels = []
        PAD = torch.tensor([0], dtype=torch.float).unsqueeze(0)
        for batch in valid_iter:
            # with AMP.autocast(enabled=self.trainer_config.fp16):
                # if self.trainer_config.swa:
                #     preds.append(self.get_logits(batch, use_swa=self.swa_flag).cpu().squeeze(-1))
                # else:
            logits = self.get_logits(batch).cpu().squeeze(-1)
            bs, length = logits.shape
            batch_pad = torch.cat([PAD] * bs, dim=0)
            logits = torch.cat([logits] + [batch_pad] * (512 - length), dim=1)
            preds.append(logits)
            # preds.append(self.get_logits(batch).cpu().squeeze(-1))
            # labels.append(batch["labels"])
        preds = torch.cat(preds, dim=0)
        # logging.debug(f"preds shape: {preds.shape}")
        # labels = torch.cat(labels, dim=0)
        # logging.debug(f"preds: {preds}")
        f1 = self.metrics(preds.numpy(), valid_datasets.samples)
        logging.info("Valid F1: {:.4f}".format(f1))
        # f1 = self.metrics(labels.numpy(), valid_datasets.samples)
        # logging.info("Label F1: {:.4f}".format(f1))
        if self.f1_maxn < f1:
            self.f1_maxn = f1
            self.save()
        del valid_iter
        gc.collect()
        self.model.train()
        if self.trainer_config.ema:
            self.ema.restore()

    def metrics(self, preds, valid_samples):
        char_probs = Utils.get_char_probs(valid_samples, preds)
        results = Utils.get_results_v2(char_probs, self.valid_text, th=0.5)
        preds = Utils.get_predictions(results)
        logging.debug(f"preds: {preds}")
        score = Utils.get_score(self.valid_labels, preds)
        
        return score

    def save(self):
        if self.trainer_config.debug:
            return
        # if self.trainer_config.swa and self.swa_flag:
        #     torch.save(self.swa_model.state_dict(), self.trainer_config.model_save)
        # else:
        torch.save(self.model.state_dict(), self.trainer_config.model_save)

    def model_load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.train()

    @torch.no_grad()
    def predict(self, test_iter):
        self.model.eval()
        logits_list = []
        for batch in test_iter:
            logits_list.append(self.get_logits(batch).view(-1).cpu())
        logits = torch.cat(logits_list, dim=-1)
        return logits.view(-1).tolist()

    def set_training_size(self, sz):
        self.training_size = self.trainer_config.epoch * sz // self.trainer_config.opt_step

    def freeze(self, layer):
        name_list = [f"layer.{i + layer}" for i in range(0, 24, 2)]
        for k, v in self.model.named_parameters():
            flag = True
            for name in name_list:
                if name in k:
                    flag = False
            v.requires_grad = flag
    
    def set_valid_datasets(self, valid_datasets, valid_collate=None):
        self.valid_datasets = valid_datasets
        self.valid_collate = valid_collate

    def set_valid_labels(self, valid_labels):
        self.valid_labels = valid_labels
    
    def set_valid_text(self, valid_text):
        self.valid_text = valid_text


class Predicter(BaseTrainer):
    def __init__(self, args):
        super(Predicter, self).__init__(args)

    @torch.no_grad()
    def predict(self, valid_datasets, valid_collate):
        self.model.eval()
        valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size, collate_fn=valid_collate)
        preds = []
        # labels = []
        PAD = torch.tensor([0], dtype=torch.float).unsqueeze(0)
        for batch in tqdm(valid_iter):
            with AMP.autocast(enabled=self.trainer_config.fp16):
                logits = self.get_logits(batch).cpu().squeeze(-1)
                bs, length = logits.shape
                batch_pad = torch.cat([PAD] * bs, dim=0)
                logits = torch.cat([logits] + [batch_pad] * (512 - length), dim=1)
                preds.append(logits)
                # labels.append(batch["labels"])
        preds = torch.cat(preds, dim=0)
        return preds


class GenTrainer(BaseTrainer):
    def __init__(self, args):
        super(GenTrainer, self).__init__(args)
    
    def build_model(self):
        self.model = Model.NBMEGenModel(self.model_config)
    
    def get_logits(self, batch, return_loss=False):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        # logging.debug(f"attention_mask: {attention_mask}")
        if return_loss:
            labels = batch["labels"].to(self.device)
            gen_labels = batch["gen_labels"].to(self.device)
            # logging.debug(f"labels: {labels.tolist()}")
            # logging.debug(f"ids: {input_ids.tolist()}")
            logits, loss = self.model(input_ids, attention_mask, labels=labels, gen_labels=gen_labels)
            return logits, loss
        else:
            logits, _ = self.model(input_ids, attention_mask)
            return logits


class GPTrainer(BaseTrainer):
    def __init__(self, args):
        super(GPTrainer, self).__init__(args)

    def build_model(self):
        self.model = Model.NBMEGPModel(self.model_config)
    
    def get_logits(self, batch, return_loss=False, use_swa=False):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        if return_loss:
            gp_labels = batch["gp_labels"].to(self.device)
            logits, loss = self.model(input_ids, attention_mask, labels=gp_labels)
            return logits, loss
        else:
            if use_swa:
                logits, _ = self.swa_model(input_ids, attention_mask)
            else:
                logits, _ = self.model(input_ids, attention_mask)
            return logits
    
    def metrics(self, preds, valid_samples):
        char_probs = Utils.get_char_probs_gp(valid_samples, preds)
        results = Utils.get_results_v2(char_probs, self.valid_text, th=0.5)
        preds = Utils.get_predictions(results)
        logging.debug(f"preds: {preds}")
        score = Utils.get_score(self.valid_labels, preds)
        return score


class CharTrainer(BaseTrainer):
    def __init__(self, args):
        super(CharTrainer, self).__init__(args)

    def build_model(self):
        self.model = Model.NBMECharModel(self.model_config)
    
    def get_logits(self, batch, return_loss=False, use_swa=False):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        # token_type_ids = batch["token_type_ids"].to(self.device)
        if return_loss:
            labels = batch["labels"].to(self.device)
            char_labels = batch["char_labels"].to(self.device)
            token_to_char_index = batch["token_to_char_index"].to(self.device)
            logits, loss = self.model(input_ids, attention_mask, labels=labels, char_labels=char_labels, token_to_char_index=token_to_char_index)
            return logits, loss
        else:
            token_to_char_index = batch["token_to_char_index"].to(self.device)
            # if use_swa:
            #     logits, _ = self.swa_model(input_ids, attention_mask, token_type_ids=token_type_ids, token_to_char_index=token_to_char_index)
            # else:
            logits, _ = self.model(input_ids, attention_mask, token_to_char_index=token_to_char_index)
            return logits
    
    def metrics(self, preds, valid_samples):
        char_probs = Utils.get_char_probs_char(valid_samples, preds)
        results = Utils.get_results_v2(char_probs, self.valid_text, th=0.5)
        preds = Utils.get_predictions(results)
        logging.debug(f"preds: {preds}")
        score = Utils.get_score(self.valid_labels, preds)
        return score


class CharPredicter(CharTrainer):
    def __init__(self, args):
        super(CharPredicter, self).__init__(args)

    @torch.no_grad()
    def predict(self, valid_datasets, valid_collate):
        self.model.eval()
        valid_iter = torch.utils.data.DataLoader(valid_datasets, batch_size=self.trainer_config.valid_batch_size, collate_fn=valid_collate)
        preds = []
        # labels = []
        PAD = torch.tensor([0], dtype=torch.float).unsqueeze(0)
        for batch in tqdm(valid_iter):
            with AMP.autocast(enabled=self.trainer_config.fp16):
                logits = self.get_logits(batch).cpu().squeeze(-1)
                preds.append(logits)
                # labels.append(batch["labels"])
        preds = torch.cat(preds, dim=0)
        return preds