import sys
sys.path.append("/users10/lyzhang/opt/tiger/NBME/src")
import tokenizer_fix
import Utils
import pandas as pd
import numpy as np
import logging
import gc
from transformers import AutoTokenizer
import optuna
logging.getLogger().setLevel(logging.INFO)


SEACHER = True
class Config():
    valid_path = "/users10/lyzhang/opt/tiger/NBME/data/sgcvs_valid"
    data_path = "/users10/lyzhang/opt/tiger/NBME/data"
    fold = 4
    fix_length = 512

args = Config()
####### INIT DATA
if SEACHER:
    features_df = pd.read_csv("/".join([args.data_path, "features.csv"]))
    patient_df = pd.read_csv("/".join([args.data_path, "patient_notes.csv"]))
    init_data = [{} for i in range(args.fold)]
    tokenizer_list = ["/users10/lyzhang/model/deberta_v3_large", "/users10/lyzhang/model/deberta_xlarge", "/users10/lyzhang/model/roberta_large"]
    for i in range(args.fold):
        valid_df = pd.read_csv(args.valid_path + f"_fold_{i}.csv")
        valid_df = valid_df.merge(features_df, on=["feature_num", "case_num"], how="left")
        valid_df = valid_df.merge(patient_df, on=["pn_num", "case_num"], how="left")
        valid_df['text_len'] = valid_df['pn_history'].apply(lambda x: len(x))
        valid_df = valid_df.sort_values(by="text_len").reset_index(drop=True)
        valid_labels = Utils.create_labels_for_scoring(valid_df)
        text = valid_df["pn_history"].tolist()
        samples_mp = dict()
        preds_mp = dict()
        for item in tokenizer_list:
            if "deberta_v" in item:
                tokenizer = tokenizer_fix.get_deberta_tokenizer(item)
            else:
                tokenizer = AutoTokenizer.from_pretrained(item, trim_offsets=False)
            valid_samples = Utils.prepare_valid_data(valid_df, tokenizer, args.fix_length)
            name = item.split("/")[-1]
            samples_mp[name] = valid_samples
            if "xlarge" in item:
                num = 2
            else:
                num = 2
            preds = [np.load("/users10/lyzhang/opt/tiger/NBME/output/"+"_".join([name, "Fold", str(i), "Id", str(idx)])+".npy") for idx in range(num)]
            preds_mp[name] = preds
        init_data[i]={
            "samples": samples_mp,
            "preds": preds_mp,
            "text": text,
            "valid_labels": valid_labels,
            "valid_df": valid_df,
        }
####### INIT DATA


@Utils.timer
def predict(trial):
    # params = [
    #     trial.suggest_int(name="deberta_v3_large_0", low=1, high=100),
    #     trial.suggest_int(name="deberta_v3_large_1", low=1, high=100),
    #     trial.suggest_int(name="deberta_v3_large_2", low=1, high=100),
    #     trial.suggest_int(name="deberta_xlarge_0", low=1, high=100),
    #     trial.suggest_int(name="deberta_xlarge_1", low=1, high=100),
    #     trial.suggest_int(name="roberta_large_0", low=1, high=100),
    #     trial.suggest_int(name="roberta_large_1", low=1, high=100),
    #     trial.suggest_int(name="roberta_large_2", low=1, high=100),
    # ]
    params = [96, 79, 40, 88, 48, 39]
    # params = [99, 69, 72, 15, 1, 2, 19, 75]
    logging.info(params)
    SUM = sum(params)
    score_list = []
    feature_gold_mp = dict()
    feature_pred_mp = dict()
    for i in range(args.fold):
        text = init_data[i]["text"]
        samples_mp = init_data[i]["samples"]
        preds_mp = init_data[i]["preds"]
        valid_labels = init_data[i]["valid_labels"]
        valid_df = init_data[i]["valid_df"]
        id_list = valid_df["id"].values.tolist()
        results = [np.zeros(len(item)) for item in text]
        top = 0
        for item in tokenizer_list:
            name = item.split("/")[-1]
            preds = preds_mp[name]
            samples = samples_mp[name]
            for p in preds:
                p = p * (params[top] / SUM)
                top += 1
                char_probs = Utils.get_char_probs(samples, p)
                for i in range(len(results)):
                    results[i] += char_probs[i]
        results = Utils.get_results_v2(results, text, th=0.5)
        # results = Utils.get_results_v3(results, text, valid_df["id"].values, th=0.5)
        preds = Utils.get_predictions(results)
        for id, p, g in zip(id_list, preds, valid_labels):
            feature_num = id.split("_")[-1]
            if feature_gold_mp.get(feature_num, None) is None:
                feature_gold_mp[feature_num] = []
            if feature_pred_mp.get(feature_num, None) is None:
                feature_pred_mp[feature_num] = []
            feature_gold_mp[feature_num].append(g)
            feature_pred_mp[feature_num].append(p)
        # score = Utils.get_score(valid_labels, preds)
        # score_list.append(score)
    data = []
    for k, pred in feature_pred_mp.items():
        gold = feature_gold_mp[k]
        num = 0
        for g in gold:
            if len(g) > 0:
                num += 1
        f1, precision, recall = Utils.get_score_all(gold, pred)
        data.append((k, precision, recall, f1, abs(recall-precision), num))
    df = pd.DataFrame(data, columns=["feature_num", "precision", "recall", "f1", "diff", "num"])
    df = df.sort_values(by="f1")
    logging.info(df.head(20))
    df.to_csv("/users10/lyzhang/opt/tiger/NBME/output/analyze_2.csv")

    # logging.info(f"score_list: {score_list}")
    # score = np.array(score_list).mean()
    # logging.info(f"mean score: {score}")
    # return score


if __name__ == "__main__":
    seacher = False
    if seacher:
        study = optuna.create_study(direction="maximize")
        study.optimize(predict, n_trials=300000, timeout=100000000)
    else:
        predict(0)