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
import pickle
logging.getLogger().setLevel(logging.INFO)


SEACHER = False
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
    with open("/users10/lyzhang/opt/tiger/NBME/output/init_data.pkl", "wb+") as f:
        pickle.dump(init_data, f)
####### INIT DATA
tokenizer_list = ["/users10/lyzhang/model/deberta_v3_large", "/users10/lyzhang/model/deberta_xlarge", "/users10/lyzhang/model/roberta_large"]
with open("/users10/lyzhang/opt/tiger/NBME/output/init_data.pkl", "rb") as f:
    init_data = pickle.load(f)

@Utils.timer
def predict(trial):
    # params = [
    #     trial.suggest_int(name="deberta_v3_large_0", low=1, high=100),
    #     trial.suggest_int(name="deberta_v3_large_1", low=1, high=100),
    #     trial.suggest_int(name="deberta_xlarge_0", low=1, high=100),
    #     trial.suggest_int(name="deberta_xlarge_1", low=1, high=100),
    #     trial.suggest_int(name="roberta_large_0", low=1, high=100),
    #     trial.suggest_int(name="roberta_large_1", low=1, high=100),
    # ]
    # params = [96, 79, 40, 88, 48, 39]
    params = [100, 81, 42, 98, 53, 37]
    # params = [99, 69, 72, 15, 1, 2, 19, 75]
    # params = [70, 10, 86, 39, 96, 1, 3, 94]
    # logging.info(params)
    SUM = sum(params)
    # th_mp = {  "103": 0.33,  "911": 0.31,  "003": 0.35,  "313": 0.35,  "904": 0.63,  "200": 0.31,  "207": 0.35,  "510": 0.69,  "508": 0.31,  "516": 0.38,  "708": 0.51,  "206": 0.32,  "702": 0.66,  "703": 0.5,  "203": 0.42,  "807": 0.5,  "812": 0.5,  "215": 0.62,  "000": 0.6,  "108": 0.41000000000000003,  "816": 0.4,  "010": 0.64,  "502": 0.5,  "815": 0.41000000000000003,  "914": 0.35000000000000003,  "112": 0.5,  "403": 0.6,  "900": 0.49,  "802": 0.5,  "915": 0.45,  "111": 0.64,  "214": 0.37,  "208": 0.5700000000000001,  "201": 0.53,  "314": 0.5,  "504": 0.5,  "505": 0.36,  "101": 0.5,  "801": 0.54,  "404": 0.43,  "513": 0.5,  "309": 0.5,  "408": 0.38,  "608": 0.5,  "012": 0.5,  "211": 0.5,  "601": 0.56,  "507": 0.4,  "512": 0.36,  "511": 0.64,  "800": 0.5,  "002": 0.39,  "210": 0.5,  "301": 0.61,  "907": 0.5,  "302": 0.4,  "100": 0.39,  "501": 0.36,  "600": 0.5,  "304": 0.5,  "804": 0.5,  "705": 0.5,  "102": 0.41000000000000003,  "811": 0.37,  "110": 0.36,  "906": 0.58,  "401": 0.5,  "500": 0.35000000000000003,  "609": 0.64,  "312": 0.63,  "006": 0.6,  "310": 0.5,  "105": 0.39,  "107": 0.35000000000000003,  "603": 0.62,  "809": 0.5,  "916": 0.39,  "913": 0.5,  "606": 0.5700000000000001,  "605": 0.45,  "213": 0.55,  "305": 0.5,  "517": 0.5,  "704": 0.35000000000000003,  "908": 0.5,  "005": 0.38,  "910": 0.5,  "514": 0.55,  "300": 0.35000000000000003,  "803": 0.56,  "706": 0.58,  "001": 0.64,  "901": 0.5,  "813": 0.45,  "204": 0.61,  "604": 0.56,  "007": 0.5,  "905": 0.36,  "607": 0.43,  "109": 0.5,  "810": 0.62,  "902": 0.4,  "405": 0.5,  "205": 0.59,  "506": 0.55,  "610": 0.45,  "806": 0.63,  "308": 0.5,  "307": 0.64,  "315": 0.5,  "306": 0.6,  "209": 0.58,  "407": 0.5,  "515": 0.5,  "212": 0.39,  "903": 0.59,  "406": 0.61,  "805": 0.5,  "106": 0.5,  "008": 0.5,  "009": 0.43,  "909": 0.42,  "912": 0.39,  "700": 0.54,  "503": 0.38,  "400": 0.54,  "817": 0.35000000000000003,  "509": 0.55 }
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
    # punish_th = {
    #     "207": 0.35,
    #     "911": 0.31,
    #     "103": 0.33,
    #     "003": 0.35,
    #     "206": 0.32,
    #     "603": 0.62,
    #     "508": 0.31,
    #     "200": 0.31,
    #     "702": 0.66,
    #     "505": 0.36,
    #     "313": 0.35,
    #     "516": 0.38,
    #     "708": 0.51,
    #     "503": 0.38,
    #     "403": 0.6,
    #     "904": 0.63,
    #     "514": 0.55,
    #     "500": 0.35,
    #     "204": 0.61,
    #     "111": 0.64,
    #     "511": 0.64,
    #     "811": 0.37,
    #     "512": 0.36,
    #     "813": 0.45,
    #     "213": 0.55,
    #     "010": 0.64,
    #     "214": 0.37,
    #     "706": 0.58,
    #     "810": 0.62,
    #     "205": 0.59,
    #     "604": 0.56,
    #     "803": 0.56,

    #     # "306": 0.6,
    #     # "801": 0.54,
    #     # "009": 0.43,
    #     # "903": 0.59,
    #     # "312": 0.63,
    #     # "607": 0.43,
    #     # "006": 0.6,
    #     # "000": 0.6,
    #     # "107": 0.35,
    #     # "307": 0.64,
    # }
    # feature_exist = [k for k, _ in th_mp.items()]
    # th_mp = {}
    # feature_num = features_df["feature_num"].values.tolist()
    # feature_num = ["0" * (3 - len(str(item))) + str(item) for item in feature_num]
    # feature_num = list(set(feature_num) - set(feature_exist))
    feature_num = [""]
    # punish_th =  {'103': 0.33, '911': 0.31, '003': 0.35, '313': 0.35, '904': 0.63, '200': 0.31, '207': 0.35, '510': 0.69, '508': 0.31, '516': 0.38, '708': 0.51, '206': 0.32, '702': 0.66, '909': 0.5, '606': 0.5, '306': 0.5, '805': 0.5, '305': 0.5, '603': 0.62, '506': 0.5, '107': 0.51, '212': 0.5, '812': 0.5, '916': 0.5, '405': 0.5, '404': 0.5, '605': 0.5, '814': 0.5, '100': 0.5, '309': 0.5, '906': 0.5, '000': 0.5, '806': 0.5, '304': 0.5, '310': 0.5, '802': 0.5, '701': 0.5, '813': 0.49, '913': 0.5, '815': 0.5, '112': 0.5, '201': 0.5, '502': 0.5, '801': 0.5, '503': 0.4, '109': 0.5, '008': 0.5, '101': 0.5, '007': 0.5, '002': 0.5, '908': 0.5, '300': 0.47000000000000003, '905': 0.5, '406': 0.5, '308': 0.5, '203': 0.5, '110': 0.48, '816': 0.5, '213': 0.51, '401': 0.5, '803': 0.54, '501': 0.5, '602': 0.5, '006': 0.5, '408': 0.5, '914': 0.5, '509': 0.5, '600': 0.5, '507': 0.5, '513': 0.5, '804': 0.5, '303': 0.5, '901': 0.5, '216': 0.5, '106': 0.5, '314': 0.5, '817': 0.36, '004': 0.5, '809': 0.5, '604': 0.5, '312': 0.5, '705': 0.5, '514': 0.5, '900': 0.49, '409': 0.5, '005': 0.5, '607': 0.5, '301': 0.51, '210': 0.5, '009': 0.43, '307': 0.5, '609': 0.5, '700': 0.5, '704': 0.5, '211': 0.5, '315': 0.5, '400': 0.54, '108': 0.5, '500': 0.5, '204': 0.5, '915': 0.5, '407': 0.5, '311': 0.5, '511': 0.5, '703': 0.5, '903': 0.5, '810': 0.5, '517': 0.5, '102': 0.5, '902': 0.5, '111': 0.53, '707': 0.5, '209': 0.5, '205': 0.5, '706': 0.49, '800': 0.5, '808': 0.5, '011': 0.5, '608': 0.5, '012': 0.5, '215': 0.5, '512': 0.5, '505': 0.51, '910': 0.5, '907': 0.5, '403': 0.5, '202': 0.5, '504': 0.5, '001': 0.5, '811': 0.5, '515': 0.5, '611': 0.5, '601': 0.5, '807': 0.5, '214': 0.5, '010': 0.5, '402': 0.5, '610': 0.5, '302': 0.48, '912': 0.5, '208': 0.5, '104': 0.5, '105': 0.5}
    # feature_num = ['207', '209', '911', '103', '003', '215', '603', '510', '809', '509', '206', '513', '402', '508', '200', '505', '504', '702', '708', '313', '516', '008', '503', '216', '514', '212', '403', '904', '204', '201', '500', '511', '111', '902', '609', '813', '517', '811', '010', '512', '203', '213', '214', '604', '706', '205', '310', '801', '306', '009', '005', '903', '803', '312', '810', '607']
    logging.info(feature_num)
    logging.info(len(feature_num))
    pre_score = 0
    # default_list = [209, 807, 911, 809, 207, 907, 215, 515, 502, 804, 910, 8, 100, 201, 206, 403, 903, 311, 310, 609, 905, 101, 510, 915, 506, 909, 5, 705, 511, 108, 505, 202, 703, 4, 404, 210, 312, 512, 305, 405, 803, 904, 608, 810, 402, 204, 106, 815, 813, 814, 514, 105, 610, 200, 704, 916, 109, 2, 508, 816, 900, 908, 7, 313, 706, 211, 306, 509, 513, 800, 812, 203, 214, 304, 307, 600, 516, 701, 806, 811]
    # logging.info(len(default_list))
    # p2r = [911, 510, 508, 200, 505, 313, 516, 201, 904, 511, 512, 813, 706, 803, 4, 608, 916, 502, 703, 2, 900, 100, 814, 910, 405]
    # r2p = [207, 215, 509, 809, 402, 513, 8, 514, 204, 811, 214, 203, 810, 310, 306, 905, 903, 312, 311, 515, 307, 109, 610, 202, 600, 506, 815, 305, 806, 211, 101]
    # o = [807, 209, 206, 403, 609, 5, 106, 915, 108, 105, 704, 701, 404, 800, 816, 909, 7, 304, 210, 812, 705, 907, 908, 804]
    # default_list = ["0" * (3 - len(str(item))) + str(item) for item in default_list]
    # p2r = ["0" * (3 - len(str(item))) + str(item) for item in p2r]
    # r2p = ["0" * (3 - len(str(item))) + str(item) for item in r2p]
    # o = ["0" * (3 - len(str(item))) + str(item) for item in o]
    # feature_num = default_list
    # for key in default_list:
        # th_mp[str(key)] = punish_th.get(str(key), 0.5)
        # th_mp[str(key)] = 0.5
    maxn_score = 0
    maxn_th = 0
    for th in range(35, 65):
        th *= 0.01
        score_list = []
        for i in range(args.fold):
            text = init_data[i]["text"]
            samples_mp = init_data[i]["samples"]
            preds_mp = init_data[i]["preds"]
            valid_labels = init_data[i]["valid_labels"]
            valid_df = init_data[i]["valid_df"]
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
            # results = Utils.get_results_v2(results, text, th=0.5)
            # results = Utils.get_results_v3(results, text, valid_df["id"].values.tolist(), th_mp, th=0.5)
            results = Utils.get_results_v2_year(results, text, th=th)
            preds = Utils.get_predictions(results)
            score = Utils.get_score(valid_labels, preds)
            score_list.append(score)
        logging.info(f"score_list: {score_list}")
        score = np.array(score_list).mean()
        logging.info(f"mean score: {score} th: {th}")
        if maxn_score < score:
            maxn_score = score
            maxn_th = th
    logging.info(f"maxn_score: {maxn_score} maxn_th: {maxn_th}")
    return score


if __name__ == "__main__":
    seacher = False
    if seacher:
        study = optuna.create_study(direction="maximize")
        study.optimize(predict, n_trials=300000, timeout=100000000)
    else:
        predict(0)