import random


with open("/users10/lyzhang/opt/tiger/NBME/language_model/tmp/tmp.txt", "r", encoding="utf-8") as f:
    data = f.readlines()
random.shuffle(data)
data_train = data[:int(len(data)*0.95)]
data_valid = data[int(len(data)*0.95):]
with open("/users10/lyzhang/opt/tiger/NBME/language_model/tmp/train.txt", "w", encoding="utf-8") as f:
    f.writelines(data_train)
with open("/users10/lyzhang/opt/tiger/NBME/language_model/tmp/valid.txt", "w", encoding="utf-8") as f:
    f.writelines(data_valid)