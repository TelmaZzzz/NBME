import pandas as pd


df = pd.read_csv("/users10/lyzhang/opt/tiger/NBME/data/patient_notes_no_train.csv")
features = pd.read_csv("/users10/lyzhang/opt/tiger/NBME/data/features.csv")
mp = dict()
for _, row in features.iterrows():
    if mp.get(row["case_num"], None) is None:
        mp[row["case_num"]] = []
    mp[row["case_num"]].append(row["feature_num"])

data = []
for _, row in df.iterrows():
    for feature_num in mp[row["case_num"]]:
        id = "_".join([str(row["pn_num"]), str(feature_num)])
        data.append((id, row["case_num"], row["pn_num"], feature_num))
output = pd.DataFrame(data, columns=["id", "case_num", "pn_num", "feature_num"])
print(output)
output.to_csv("/users10/lyzhang/opt/tiger/NBME/data/no_labels.csv", index=False)
