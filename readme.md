## Kaggle NBME - Score Clinical Patient Notes
---
Public Score: 0.894(rank 5/1500)

Private Score: 0.892(rank 26/1500)

* 主干部分：roberta-large deberta-v3-large deberta-xlarge 加权融合
* roberta-large 用MLM进行预训练
* 3类模型都进行一轮伪标签训练
* Fine-tune过程使用FGM、EMA、5dropout

Inference Code: [https://www.kaggle.com/code/telmazzzz/nbme-inference](https://www.kaggle.com/code/telmazzzz/nbme-inference)

Train: `./script`