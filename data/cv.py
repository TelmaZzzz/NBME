import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


df = pd.read_csv("/users10/lyzhang/opt/tiger/NBME/data/pseudo/submission_1.csv")
sgcv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=959)
for fold, (train_ids, valid_ids) in enumerate(sgcv.split(df, df["case_num"], df["pn_num"])):
    train_df = df.iloc[train_ids]
    valid_df = df.iloc[valid_ids]
    print(f"train size: {len(train_df)}")
    print(f"valid size: {len(valid_df)}")
    # train_df.to_csv(f"sgcvs_train_fold_{fold}.csv", index=False)
    valid_df.to_csv(f"/users10/lyzhang/opt/tiger/NBME/data/pseudo/submission_1_fold_{fold}.csv", index=False)
