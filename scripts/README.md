We provide detailed hyper-parameters to reproduce the experimental results in this file.

### Main Experiment
1. Train model on DBpedia15K, we denote the path for saved model as MODEL_PATH1.
> python train.py --model_path none --kg1 DBpedia15K --kg0 none --embedding_lr 2e-3 --adapter_lr 1e-3 --alpha 1e4 --beta -1.0 --epoch 20 --device cuda:0 --bert_path BERT_PATH

2. Train model on Wikidata15K with forward knowledge transfer from DBpedia15K, we denote the path for saved model as MODEL_PATH2.
> python train.py --model_path MODEL_PATH1 --kg1 Wikidata15K --kg0 DBpedia15K --embedding_lr 1e-3 --adapter_lr 1e-3 --alpha 1e4 --beta -1.0 --epoch 40 --device cuda:0 --bert_path BERT_PATH

3. Train model on YAGO15K with forward knowledge transfer from Wikidata15K
> python train.py --model_path MODEL_PATH2 --kg1 Yago15K --kg0 Wikidata15K --embedding_lr 2e-3 --adapter_lr 1e-3 --alpha 1e4 --beta -1.0 --epoch 20 --device cuda:0 --bert_path BERT_PATH
