from __future__ import annotations
import pandas as pd


df = pd.read_csv("/users10/lyzhang/opt/tiger/NBME/tmp.csv")
# df_f = pd.read_csv("/users10/lyzhang/opt/tiger/NBME/data/features.csv")
# df_p = pd.read_csv("/users10/lyzhang/opt/tiger/NBME/data/patient_notes.csv")
# df = df.merge(df_f, on=['feature_num', 'case_num'], how='left')
# df = df.merge(df_p, on=['pn_num', 'case_num'], how='left')
# df.to_csv("/users10/lyzhang/opt/tiger/NBME/tmp.csv", index=False)
# df = df.sample(5)
# tmp = df[df["id"]=="00082_009"]
cnt = 0
index_list = []
for idx, row in df.iterrows():
    if ";" in row["location"]:
        cnt += 1
        index_list.append(idx)
df_ = df.iloc[index_list]
df_ = df_.sample(15, random_state=42)
for idx, row in df_.iterrows():
    pn_history = row["pn_history"]
    location = eval(row["location"])
    annotations = eval(row["annotation"])
    for lo, an in zip(location, annotations):
        lo = lo.split(" ")
        l = lo[0]
        r = lo[-1]
        print(f"pn: {pn_history[int(l): int(r)]}")
        print(f"an: {row['annotation']}")
    # print(row["location"])
    # print(row["annotation"])
    # print(pn_history[512: 542])
