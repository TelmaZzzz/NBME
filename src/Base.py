import tokenizer_fix
import os
import logging
import datetime
import torch
import Config, Datasets, Model, Trainer, Utils
from transformers import AutoTokenizer
import pandas as pd
import time
import gc
from tqdm import tqdm
import copy
import numpy as np


@Utils.timer
def NBMEBase(args):
    train_df = pd.read_csv(args.train_path + f"_fold_{args.fold}.csv")
    valid_df = pd.read_csv(args.valid_path + f"_fold_{args.fold}.csv")
    features_df = pd.read_csv("/".join([args.data_path, "features.csv"]))
    patient_df = pd.read_csv("/".join([args.data_path, "patient_notes.csv"]))
    train_df = train_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    train_df = train_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    valid_df = valid_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    valid_df = valid_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    if args.debug:
        train_df = train_df.sample(100)
        valid_df = train_df
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    if args.deberta:
        tokenizer = tokenizer_fix.get_deberta_tokenizer(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trim_offsets=False)
    if args.offset_fix:
        train_samples = Utils.prepare_training_data_fix(train_df, tokenizer, args.fix_length)
        valid_samples = Utils.prepare_training_data_fix(valid_df, tokenizer, args.fix_length)
    else:
        train_samples = Utils.prepare_training_data(train_df, tokenizer, args.fix_length)
        valid_samples = Utils.prepare_valid_data(valid_df, tokenizer, args.fix_length)
        # valid_samples = Utils.prepare_training_data(valid_df, tokenizer, args.fix_length)
    if args.da:
        da_df = pd.read_csv(args.da_path)
        da_df = da_df.merge(features_df, on=["feature_num", "case_num"], how="left")
        da_df = da_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
        da_df = da_df.fillna(value="")
        if args.debug:
            da_df = da_df.sample(100)
        da_samples = Utils.prepare_training_data_da(da_df, tokenizer, args.fix_length)
        train_samples.extend(da_samples)
    train_datasets = Datasets.NBMEDataset(train_samples, tokenizer, mask_prob=args.mask_prob, mask_ratio=args.mask_ratio)
    # valid_datasets = Datasets.NBMEDataset(valid_samples, tokenizer)
    valid_datasets = Datasets.NBMEDatasetValid(valid_samples, tokenizer)
    valid_collate = Datasets.Collate(tokenizer)
    valid_labels = Utils.create_labels_for_scoring(valid_df)
    valid_text = valid_df["pn_history"].tolist()
    # logging.debug(f"valid_labels: {valid_labels}")
    # logging.debug(f"{valid_df['id'].unique().tolist()}")
    # for a, b in zip(valid_df['id'].unique().tolist(), valid_labels):
    #     logging.debug(a)
    #     logging.debug(b)
    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True)
    trainer = Trainer.BaseTrainer(args)
    trainer.set_training_size(len(train_iter))
    trainer.model_init()
    trainer.optimizer_init()
    # trainer.set_valid_datasets(valid_datasets)
    trainer.set_valid_datasets(valid_datasets, valid_collate)
    trainer.set_valid_labels(valid_labels)
    trainer.set_valid_text(valid_text)
    if args.model_load is not None:
        trainer.model_load(args.model_load)
    logging.info(f"Train Size: {len(train_iter)}")
    for epoch in range(args.epoch):
        logging.info(f"Start Epoch: {epoch}")
        # if epoch >= 8:
        #     continue
        t_s = time.time()
        loss = 0
        if args.debug:
            for batch in tqdm(train_iter):
                loss += trainer.step(batch)
        else:
            for batch in train_iter:
                loss += trainer.step(batch)
        logging.info("Train Loss: {:.4f}".format(loss / len(train_iter)))
        t_e = time.time()
        logging.info("Cost {:.2f} s.".format(t_e - t_s))
    f1_maxn = trainer.f1_maxn
    logging.info("Best F1: {:.4f}".format(f1_maxn))
    del train_df, valid_df
    del train_samples, valid_samples
    del train_datasets, valid_datasets
    del train_iter, trainer
    gc.collect()
    return f1_maxn


@Utils.timer
def NBMEGenerate(args):
    train_df = pd.read_csv(args.train_path + f"_fold_{args.fold}.csv")
    valid_df = pd.read_csv(args.valid_path + f"_fold_{args.fold}.csv")
    features_df = pd.read_csv("/".join([args.data_path, "features.csv"]))
    patient_df = pd.read_csv("/".join([args.data_path, "patient_notes.csv"]))
    train_df = train_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    train_df = train_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    valid_df = valid_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    valid_df = valid_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    # if args.debug:
    #     train_df = train_df.sample(100)
    #     valid_df = train_df
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    if args.deberta:
        tokenizer = tokenizer_fix.get_deberta_tokenizer(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trim_offsets=False)
    train_samples = Utils.prepare_training_data_gen(train_df, tokenizer, args.fix_length)
    valid_samples = Utils.prepare_training_data_gen(valid_df, tokenizer, args.fix_length)
    train_datasets = Datasets.NBMEGenDataset(train_samples, tokenizer, mask_prob=args.mask_prob, mask_ratio=args.mask_ratio)
    valid_datasets = Datasets.NBMEGenDataset(valid_samples, tokenizer)
    valid_labels = Utils.create_labels_for_scoring(valid_df)
    valid_text = valid_df["pn_history"].tolist()
    # logging.debug(f"valid_labels: {valid_labels}")
    # logging.debug(f"{valid_df['id'].unique().tolist()}")
    # for a, b in zip(valid_df['id'].unique().tolist(), valid_labels):
    #     logging.debug(a)
    #     logging.debug(b)
    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True)
    trainer = Trainer.GenTrainer(args)
    trainer.set_training_size(len(train_iter))
    trainer.model_init()
    trainer.optimizer_init()
    trainer.set_valid_datasets(valid_datasets)
    trainer.set_valid_labels(valid_labels)
    trainer.set_valid_text(valid_text)
    logging.info(f"Train Size: {len(train_iter)}")
    for epoch in range(args.epoch):
        logging.info(f"Start Epoch: {epoch}")
        # if epoch >= 8:
        #     continue
        t_s = time.time()
        loss = 0
        if args.debug:
            for batch in tqdm(train_iter):
                loss += trainer.step(batch)
        else:
            for batch in train_iter:
                loss += trainer.step(batch)
        logging.info("Train Loss: {:.4f}".format(loss / len(train_iter)))
        t_e = time.time()
        logging.info("Cost {:.2f} s.".format(t_e - t_s))
    f1_maxn = trainer.f1_maxn
    logging.info("Best F1: {:.4f}".format(f1_maxn))
    del train_df, valid_df
    del train_samples, valid_samples
    del train_datasets, valid_datasets
    del train_iter, trainer
    gc.collect()
    return f1_maxn

@Utils.timer
def NBMEGP(args):
    train_df = pd.read_csv(args.train_path + f"_fold_{args.fold}.csv")
    valid_df = pd.read_csv(args.valid_path + f"_fold_{args.fold}.csv")
    features_df = pd.read_csv("/".join([args.data_path, "features.csv"]))
    patient_df = pd.read_csv("/".join([args.data_path, "patient_notes.csv"]))
    train_df = train_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    train_df = train_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    valid_df = valid_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    valid_df = valid_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    if args.debug:
        train_df = train_df.sample(100)
        valid_df = train_df
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    if args.deberta:
        tokenizer = tokenizer_fix.get_deberta_tokenizer(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trim_offsets=False)
    if args.offset_fix:
        train_samples = Utils.prepare_training_data_fix(train_df, tokenizer, args.fix_length)
        valid_samples = Utils.prepare_training_data_fix(valid_df, tokenizer, args.fix_length)
    else:
        train_samples = Utils.prepare_training_data(train_df, tokenizer, args.fix_length)
        valid_samples = Utils.prepare_training_data(valid_df, tokenizer, args.fix_length)
    train_datasets = Datasets.NBMEGPDataset(train_samples, tokenizer, mask_prob=args.mask_prob, mask_ratio=args.mask_ratio)
    valid_datasets = Datasets.NBMEGPDataset(valid_samples, tokenizer)
    valid_labels = Utils.create_labels_for_scoring(valid_df)
    valid_text = valid_df["pn_history"].tolist()
    # logging.debug(f"valid_labels: {valid_labels}")
    # logging.debug(f"{valid_df['id'].unique().tolist()}")
    # for a, b in zip(valid_df['id'].unique().tolist(), valid_labels):
    #     logging.debug(a)
    #     logging.debug(b)
    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True)
    trainer = Trainer.GPTrainer(args)
    trainer.set_training_size(len(train_iter))
    trainer.model_init()
    trainer.optimizer_init()
    trainer.set_valid_datasets(valid_datasets)
    trainer.set_valid_labels(valid_labels)
    trainer.set_valid_text(valid_text)
    logging.info(f"Train Size: {len(train_iter)}")
    for epoch in range(args.epoch):
        logging.info(f"Start Epoch: {epoch}")
        # if epoch >= 8:
        #     continue
        t_s = time.time()
        loss = 0
        if args.debug:
            for batch in tqdm(train_iter):
                loss += trainer.step(batch)
        else:
            for batch in train_iter:
                loss += trainer.step(batch)
        logging.info("Train Loss: {:.4f}".format(loss / len(train_iter)))
        t_e = time.time()
        logging.info("Cost {:.2f} s.".format(t_e - t_s))
    f1_maxn = trainer.f1_maxn
    logging.info("Best F1: {:.4f}".format(f1_maxn))
    del train_df, valid_df
    del train_samples, valid_samples
    del train_datasets, valid_datasets
    del train_iter, trainer
    gc.collect()
    return f1_maxn


@Utils.timer
def NBMEChar(args):
    train_df = pd.read_csv(args.train_path + f"_fold_{args.fold}.csv")
    valid_df = pd.read_csv(args.valid_path + f"_fold_{args.fold}.csv")
    features_df = pd.read_csv("/".join([args.data_path, "features.csv"]))
    patient_df = pd.read_csv("/".join([args.data_path, "patient_notes.csv"]))
    train_df = train_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    train_df = train_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    valid_df = valid_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    valid_df = valid_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    if args.debug:
        train_df = train_df.sample(100)
        valid_df = train_df
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    if args.deberta:
        tokenizer = tokenizer_fix.get_deberta_tokenizer(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trim_offsets=False)
    train_samples = Utils.prepare_training_data_char(train_df, tokenizer, args.fix_length)
    valid_samples = Utils.prepare_training_data_char(valid_df, tokenizer, args.fix_length)
    train_datasets = Datasets.NBMECharDataset(train_samples, tokenizer, mask_prob=args.mask_prob, mask_ratio=args.mask_ratio)
    valid_datasets = Datasets.NBMECharDataset(valid_samples, tokenizer)
    valid_labels = Utils.create_labels_for_scoring(valid_df)
    valid_text = valid_df["pn_history"].tolist()
    # logging.debug(f"valid_labels: {valid_labels}")
    # logging.debug(f"{valid_df['id'].unique().tolist()}")
    # for a, b in zip(valid_df['id'].unique().tolist(), valid_labels):
    #     logging.debug(a)
    #     logging.debug(b)
    train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True)
    trainer = Trainer.CharTrainer(args)
    trainer.set_training_size(len(train_iter))
    trainer.model_init()
    trainer.optimizer_init()
    trainer.set_valid_datasets(valid_datasets)
    trainer.set_valid_labels(valid_labels)
    trainer.set_valid_text(valid_text)
    logging.info(f"Train Size: {len(train_iter)}")
    for epoch in range(args.epoch):
        logging.info(f"Start Epoch: {epoch}")
        if epoch >= 10:
            continue
        t_s = time.time()
        loss = 0
        if args.debug:
            for batch in tqdm(train_iter):
                loss += trainer.step(batch)
        else:
            for batch in train_iter:
                loss += trainer.step(batch)
        logging.info("Train Loss: {:.4f}".format(loss / len(train_iter)))
        t_e = time.time()
        logging.info("Cost {:.2f} s.".format(t_e - t_s))
    f1_maxn = trainer.f1_maxn
    logging.info("Best F1: {:.4f}".format(f1_maxn))
    del train_df, valid_df
    del train_samples, valid_samples
    del train_datasets, valid_datasets
    del train_iter, trainer
    gc.collect()
    return f1_maxn


@Utils.timer
def main(args):
    if args.mode not in ["base", "generate", "gp", "char"]:
        raise
    model_save = "/".join([args.model_save, Utils.d2s(datetime.datetime.now(), time=True)])
    if not args.debug:
        if os.path.exists(model_save):
            logging.warning("save path exists, sleep 60s")
            raise
        else:
            os.mkdir(model_save)
            args.model_save = model_save
    MODEL_PREFIX = args.model_save
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"device: {args.device}")
    if args.train_all:
        num = args.fold
        for fold in range(num):
            args.fold = fold
            args.model_save = "/".join([MODEL_PREFIX, f"Fold_{args.fold}.bin"])
            logging.info(f"model save path: {args.model_save}")
            if args.mode == "base":
                NBMEBase(args)
            elif args.mode == "generate":
                NBMEGenerate(args)
    else:
        args.model_save = "/".join([MODEL_PREFIX, f"Fold_{args.fold}.bin"])
        logging.info(f"model save path: {args.model_save}")
        if args.mode == "base":
            NBMEBase(args)
        elif args.mode == "generate":
            NBMEGenerate(args)
        elif args.mode == "gp":
            NBMEGP(args)
        elif args.mode == "char":
            NBMEChar(args)


@Utils.timer
def predict(args):
    score = True
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # th_mp = {
    #     '103': 0.33,
    #     '911': 0.31,
    #     '003': 0.35,
    #     '313': 0.35,
    #     '904': 0.63,
    #     '200': 0.31,
    #     '207': 0.35,
    #     '510': 0.69,
    #     '508': 0.31,
    #     "516": 0.38,
    #     "708": 0.51,
    #     "206": 0.32,
    #     "702": 0.66,
    # }
    # params = [96, 79, 40, 88, 48, 39]
    model_list = [
        # {
        #     "model_path": [
        #         # # deberta-v3-large
        #         # # no pseudo
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_02_13_17/Fold_0.bin", params[0]),
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_13_16_58_05/Fold_1.bin", params[0]), # 0.8873
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_14_12_36_34/Fold_2.bin", params[0]), # 0.8889
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_01_13_40/Fold_3.bin", params[0]),
        #         # # pseduo
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_15_12_30_33/Fold_0.bin", params[1]), # 89.63
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_19_17_46_56/Fold_1.bin", params[1]), # 89.11
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_17_13_25_39/Fold_2.bin", params[1]), # 89.06
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_16_23_09_58/Fold_3.bin", params[1]), # 89.72
        #         # # pseduo round 2
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_28_19_36_40/Fold_0.bin", 1), # 89.65
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_29_13_53_35/Fold_1.bin", 1), # 89.27
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_28_15_46_17/Fold_2.bin", 1), # 89.33
        #         ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_29_00_08_00/Fold_3.bin", 1), # 89.56
        #     ],
        #     "pretrain_path": "/users10/lyzhang/model/deberta_v3_large",
        #     "mode": "base",
        # },
        # {
        #     "model_path": [
        #         # # deberta-xlarge
        #         # # no pseudo
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_03_31_21_38/Fold_0.bin", params[2]),
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_02_03_10/Fold_1.bin", params[2]),
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_03_31_21_38/Fold_2.bin", params[2]),
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_03_31_21_38/Fold_3.bin", params[2]),
        #         # pseudo
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_21_14_53_32/Fold_0.bin", params[3]), # 89.50
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_21_17_58_47/Fold_1.bin", params[3]), # 89.03
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_23_11_32_56/Fold_2.bin", params[3]), # 88.96
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_21_17_57_19/Fold_3.bin", params[3]), #89.56
        #     ],
        #     "pretrain_path": "/users10/lyzhang/model/deberta_xlarge",
        #     "mode": "base",
        # },
        # {
        #     "model_path": [
        #         # # roberta-large
        #         # no pseudo
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_13_00_25_39/Fold_0.bin", params[4]),
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_13_00_25_39/Fold_1.bin", params[4]),
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_13_00_25_39/Fold_2.bin", params[4]),
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_13_00_25_39/Fold_3.bin", params[4]),
        #         # pseudo
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_15_19_09_26/Fold_0.bin", params[5]), # 89.34
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_17_12_22_35/Fold_1.bin", params[5]), # 89.06
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_17_12_23_13/Fold_2.bin", params[5]), # 89.00
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_17_00_27_04/Fold_3.bin", params[5]), # 89.49
        #         # pseudo round 2
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_28_15_47_58/Fold_0.bin", 1), # 89.54
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_28_15_48_20/Fold_1.bin", 1), # 89.22
        #         # ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_28_15_51_59/Fold_2.bin", 1), # 89.23
        #         ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_28_15_52_27/Fold_3.bin", 1), # 89.49
        #     ],
        #     "pretrain_path": "/users10/lyzhang/model/roberta_large",
        #     "mode": "base",
        # },
        {
            "model_path": [
                ("/users10/lyzhang/opt/tiger/NBME/model/Base/2022_04_05_00_12/Fold_1.bin", 1),
            ],
            "pretrain_path": "/users10/lyzhang/model/deberta_v2_xlarge",
            "mode": "base",
        },
    ]
    model_num = 0
    for model_config in model_list:
        for model in model_config["model_path"]:
            model_num += model[1]
    features_df = pd.read_csv("/".join([args.data_path, "features.csv"]))
    patient_df = pd.read_csv("/".join([args.data_path, "patient_notes.csv"]))
    if score:
        valid_df = pd.read_csv(args.valid_path + f"_fold_{args.fold}.csv")
    else:
        valid_df = pd.read_csv(args.valid_path)
        # valid_df = valid_df.sample(100)
    valid_df = valid_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    valid_df = valid_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    valid_df['text_len'] = valid_df['pn_history'].apply(lambda x: len(x))
    valid_df = valid_df.sort_values(by="text_len").reset_index(drop=True)
    if score:
        valid_labels = Utils.create_labels_for_scoring(valid_df)
    text = valid_df["pn_history"].tolist()
    results = [np.zeros(len(item)) for item in text]
    for model_config in model_list:
        model_path = model_config["model_path"]
        pretrain_path = model_config["pretrain_path"]
        mode = model_config["mode"]
        args.pretrain_path = pretrain_path
        args.tokenizer_path = pretrain_path
        if mode == "base":
            predicter = Trainer.Predicter(args)
        elif mode == "char":
            predicter = Trainer.CharPredicter(args)
        predicter.model_init()
        for idx, path in enumerate(model_path):
            if "deberta_v" in args.tokenizer_path:
                tokenizer = tokenizer_fix.get_deberta_tokenizer(args.tokenizer_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trim_offsets=False)
            if score:
                valid_samples = Utils.prepare_valid_data(valid_df, tokenizer, args.fix_length)
            else:
                valid_samples = Utils.prepare_test_data(valid_df, tokenizer, args.fix_length)
            if mode == "base":
                valid_datasets = Datasets.NBMEDatasetValid(valid_samples, tokenizer)
                valid_collate = Datasets.Collate(tokenizer)
            elif mode == "char":
                valid_datasets = Datasets.NBMECharDatasetValid(valid_samples, tokenizer)
                valid_collate = Datasets.CollateChar(tokenizer)
            predicter.model_load(path[0])
            preds = predicter.predict(valid_datasets, valid_collate)
            
            np.save("/users10/lyzhang/opt/tiger/NBME/output/"+"_".join([pretrain_path.split("/")[-1], "Fold", str(args.fold), "Id", str(0)])+".npy", preds.numpy())
            preds = preds * (path[1] / model_num)
            if mode == "base":
                char_probs = Utils.get_char_probs(valid_samples, preds)
            elif mode == "char":
                char_probs = Utils.get_char_probs_char(valid_samples, preds)
            for i in range(len(results)):
                results[i] += char_probs[i]
                # logging.info(f"{len(results[i])} {len(char_probs[i])}")
            torch.cuda.empty_cache()
            del valid_samples, valid_datasets
            gc.collect()
        del predicter.model
        del predicter
        gc.collect()
    # results = Utils.get_results(char_probs, th=0.5)
    results = Utils.get_results_v2(results, text, th=0.5)
    # results = Utils.get_results_v3(results, text, valid_df["id"].values.tolist(), th_mp, th=0.5)
    if score:
        preds = Utils.get_predictions(results)
        score = Utils.get_score(valid_labels, preds)
        logging.info("f1: {:.4f}".format(score))
    else:
        submission = valid_df
        submission["location"] = results
        submission[['id', 'case_num', 'pn_num', 'feature_num', 'location']].to_csv(f'/users10/lyzhang/opt/tiger/NBME/data/pseudo/submission_round2_{args.fold}.csv', index=False)


if __name__ == "__main__":
    args = Config.BaseConfig()
    Utils.set_seed(args.seed)
    if not args.debug:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.train:
        logging.info(f"args: {args}".replace(" ", "\n"))
        main(args)
    elif args.predict:
        predict(args)