We provide detailed hyper-parameters to reproduce the experimental results in this file.

### Main Experiment
We denote the paths for best models saved in `Step{K}` as `MODEL_PATH{K}`.

`Step1` Train the model on DBpedia15K without knowledge transfer.
> python train.py --model_path none --kg1 DBpedia15K --kg0 none --embedding_lr 2e-3 --adapter_lr 1e-3 --alpha 1e4 --beta -1.0 --epoch 20 --device cuda:0 --bert_path BERT_PATH

`Step2` Train the model on Wikidata15K with forward knowledge transfer from DBpedia15K.
> python train.py --model_path MODEL_PATH1 --kg1 Wikidata15K --kg0 DBpedia15K --embedding_lr 1e-3 --adapter_lr 1e-3 --alpha 1e4 --beta -1.0 --epoch 40 --device cuda:0 --bert_path BERT_PATH

`Step3` Train the model on Wikidata15K with backward knowledge transfer to improve the performance on DBpedia15K.
> python train.py --model_path MODEL_PATH2 --kg1 Wikidata15K --kg0 DBpedia15K --embedding_lr 5e-4 --adapter_lr 1e-4 --alpha -1.0 --beta 1.0 --epoch 10 --device cuda:0 --bert_path BERT_PATH

`Step4` Train the model on YAGO15K with forward knowledge transfer from Wikidata15K.
> python train.py --model_path MODEL_PATH3 --kg1 Yago15K --kg0 Wikidata15K --embedding_lr 2e-3 --adapter_lr 1e-3 --alpha 1e4 --beta -1.0 --epoch 20 --device cuda:0 --bert_path BERT_PATH

`Step5` Train the model on YAGO15K with backward knowledge transfer to improve the performance on Wikidata15K.
> python train.py --model_path MODEL_PATH4 --kg1 Yago15K --kg0 Wikidata15K --embedding_lr 1e-3 --adapter_lr 1e-4 --alpha -1.0 --beta 1.0 --epoch 10 --device cuda:0 --bert_path BERT_PATH


### Extend to FB15K-237
`DWYF` Train the model on FB15K-237 with forward knowledge transfer from YAGO15K.
> python train.py --model_path DYW_MODEL --kg1 FB15K237 --kg0 Yago15K --embedding_lr 2e-3 --adapter_lr 1e-3 --alpha 10 --beta -1.0 --epoch 20 --device cuda:0 --bert_path BERT_PATH

For `DYWF`, `WDYF`, `WYDF`, `YDWF` and `YWDF`, change the hyperparameter $\alpha$ to 1, 10, 1, 1 and 2 respectively.
