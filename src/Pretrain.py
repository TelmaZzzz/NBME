import pandas as pd
import argparse

import warnings
warnings.filterwarnings('ignore')

from transformers import (AutoModel,AutoModelForMaskedLM, 
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, required=True)
parser.add_argument("--pretrain_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--valid_path", type=str, required=True)

args = parser.parse_args()

# data = pd.read_csv(args.train_path)
# data["excerpt"] = data["pn_history"].apply(lambda x: x.replace("\n", "").replace("\r", "").replace("\t", ""))

# text = "\n".join(data.excerpt.tolist())

# with open("/users10/lyzhang/opt/tiger/NBME/data/tmp.txt", "w") as f:
#     f.write(text)

model_name = args.pretrain_path.split("/")[-1]

model = AutoModelForMaskedLM.from_pretrained(args.pretrain_path)
tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path)
tokenizer.save_pretrained("/".join([args.save_path, model_name]));

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.train_path, #mention train text file here
    block_size=512)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.valid_path, #mention valid text file here
    block_size=512)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="/".join([args.save_path, model_name + "_ckpt"]), #select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=8,
    evaluation_strategy= 'steps',
    save_total_limit=2,
    eval_steps=2000,
    save_steps=6000,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end =True,
    prediction_loss_only=True,
    report_to = "none")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)


trainer.train()
trainer.save_model("/".join([args.save_path, model_name]))