import sys
sys.path.append("/users10/lyzhang/opt/tiger/NBME/src")
import tokenizer_fix
import Utils
import pandas as pd
import numpy as np
import logging
import gc
from transformers import AutoTokenizer
logging.getLogger().setLevel(logging.INFO)


SEACHER = False
class Config():
    valid_path = "/users10/lyzhang/opt/tiger/NBME/data/sgcvs_valid"
    data_path = "/users10/lyzhang/opt/tiger/NBME/data"
    fold = 1
    fix_length = 512

args = Config()
####### INIT DATA
# if SEACHER:
#     features_df = pd.read_csv("/".join([args.data_path, "features.csv"]))
#     patient_df = pd.read_csv("/".join([args.data_path, "patient_notes.csv"]))
#     init_data = [{} for i in range(args.fold)]
#     for i in range(args.fold):
#         valid_df = pd.read_csv(args.valid_path + f"_fold_{args.fold}.csv")
#         valid_df = valid_df.merge(features_df, on=["feature_num", "case_num"], how="left")
#         valid_df = valid_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
#         valid_df['text_len'] = valid_df['pn_history'].apply(lambda x: len(x))
#         valid_df = valid_df.sort_values(by="text_len").reset_index(drop=True)
#         valid_labels = Utils.create_labels_for_scoring(valid_df)
#         text = valid_df["pn_history"].tolist()
#         results = [np.zeros(len(item)) for item in text]


####### INIT DATA


@Utils.timer
def predict():
    model_list = [
        {
            "model_path": [
                # # deberta-v3-large
                # # no pseudo
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_v3_large_Fold_0_Id_0.npy", 1),
                ("/users10/lyzhang/opt/tiger/NBME/output/deberta_v3_large_Fold_1_Id_0.npy", 1), # 88.73
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_v3_large_Fold_2_Id_0.npy", 1), # 88.89
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_v3_large_Fold_3_Id_0.npy", 1),
                # # pseduo
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_v3_large_Fold_0_Id_1.npy", 1), # 89.63
                ("/users10/lyzhang/opt/tiger/NBME/output/deberta_v3_large_Fold_1_Id_1.npy", 1), # 89.11
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_v3_large_Fold_2_Id_1.npy", 1), # 89.06
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_v3_large_Fold_3_Id_1.npy", 1), # 89.72
            ],
            "pretrain_path": "/users10/lyzhang/model/deberta_v3_large",
            "mode": "base",
        },
        {
            "model_path": [
                # # deberta-xlarge
                # # no pseudo
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_xlarge_Fold_0_Id_0.npy", 1),
                ("/users10/lyzhang/opt/tiger/NBME/output/deberta_xlarge_Fold_1_Id_0.npy", 1),
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_xlarge_Fold_2_Id_0.npy", 1),
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_xlarge_Fold_3_Id_0.npy", 1),
                # pseudo
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_xlarge_Fold_0_Id_1.npy", 1), # 89.50
                ("/users10/lyzhang/opt/tiger/NBME/output/deberta_xlarge_Fold_1_Id_1.npy", 1), # 89.03
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_xlarge_Fold_2_Id_1.npy", 1), # 88.96
                # ("/users10/lyzhang/opt/tiger/NBME/output/deberta_xlarge_Fold_3_Id_1.npy", 1), # 89.56
            ],
            "pretrain_path": "/users10/lyzhang/model/deberta_xlarge",
            "mode": "base",
        },
        {
            "model_path": [
                # # roberta-large
                # no pseudo
                # ("/users10/lyzhang/opt/tiger/NBME/output/roberta_large_Fold_0_Id_0.npy", 1),
                ("/users10/lyzhang/opt/tiger/NBME/output/roberta_large_Fold_1_Id_0.npy", 1),
                # ("/users10/lyzhang/opt/tiger/NBME/output/roberta_large_Fold_2_Id_0.npy", 1),
                # ("/users10/lyzhang/opt/tiger/NBME/output/roberta_large_Fold_3_Id_0.npy", 1),
                # pseudo
                # ("/users10/lyzhang/opt/tiger/NBME/output/roberta_large_Fold_0_Id_1.npy", 1), # 89.34
                ("/users10/lyzhang/opt/tiger/NBME/output/roberta_large_Fold_1_Id_1.npy", 1), # 89.06
                # ("/users10/lyzhang/opt/tiger/NBME/output/roberta_large_Fold_2_Id_1.npy", 1), # 89.00
                # ("/users10/lyzhang/opt/tiger/NBME/output/roberta_large_Fold_3_Id_1.npy", 1), # 89.49
            ],
            "pretrain_path": "/users10/lyzhang/model/roberta_large",
            "mode": "base",
        },
    ]
    model_num = 0
    for model_config in model_list:
        for model in model_config["model_path"]:
            model_num += model[1]
    features_df = pd.read_csv("/".join([args.data_path, "features.csv"]))
    patient_df = pd.read_csv("/".join([args.data_path, "patient_notes.csv"]))
    valid_df = pd.read_csv(args.valid_path + f"_fold_{args.fold}.csv")
        # valid_df = valid_df.sample(100)
    valid_df = valid_df.merge(features_df, on=["feature_num", "case_num"], how="left")
    valid_df = valid_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
    valid_df['text_len'] = valid_df['pn_history'].apply(lambda x: len(x))
    valid_df = valid_df.sort_values(by="text_len").reset_index(drop=True)
    valid_labels = Utils.create_labels_for_scoring(valid_df)
    text = valid_df["pn_history"].tolist()
    results = [np.zeros(len(item)) for item in text]
    for model_config in model_list:
        model_path = model_config["model_path"]
        pretrain_path = model_config["pretrain_path"]
        args.pretrain_path = pretrain_path
        args.tokenizer_path = pretrain_path
        if "deberta_v" in args.tokenizer_path:
            tokenizer = tokenizer_fix.get_deberta_tokenizer(args.tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trim_offsets=False)
        valid_samples = Utils.prepare_valid_data(valid_df, tokenizer, args.fix_length)
        for idx, path in enumerate(model_path):
            preds = np.load("/users10/lyzhang/opt/tiger/NBME/output/"+"_".join([pretrain_path.split("/")[-1], "Fold", str(args.fold), "Id", str(idx)])+".npy")
            preds = preds * (path[1] / model_num)
            char_probs = Utils.get_char_probs(valid_samples, preds)
            for i in range(len(results)):
                results[i] += char_probs[i]
                # logging.info(f"{len(results[i])} {len(char_probs[i])}")
        del valid_samples
        gc.collect()
    # results = Utils.get_results(char_probs, th=0.5)
    # results = Utils.get_results_v2(results, text, th=0.5)
    results = Utils.get_results_v3(results, text, valid_df["id"].values.tolist(), th=0.5)
    preds = Utils.get_predictions(results)
    score = Utils.get_score(valid_labels, preds)
    # submission = valid_df
    # submission["location"] = results
    # submission[['id', 'location']].to_csv(f'/users10/lyzhang/opt/tiger/NBME/output/submission_{args.fold}.csv', index=False)
    logging.info("f1: {:.6f}".format(score))


if __name__ == "__main__":
    predict()