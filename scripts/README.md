We provide detailed hyper-parameters to reproduce the experimental results in this file.
The pre-trained language model we used is downloaded at [here](https://huggingface.co/bert-base-cased/tree/main).

### Main Experiment
We denote the paths for best models saved in `Step{K}` as `MODEL_PATH{K}`.
We set the `adapter_hidden_size` to 128, 'ea_t` to 0.999, `batch_size` to 512, `early_stop` to 5, `max_seq_length` to 64, and 'label_smoothing' to 0.8.

`Step1` Train the model on DBpedia15K without knowledge transfer.
> python train.py --kg1 DBpedia15K --embedding_lr 2e-3 --adapter_lr 1e-3 --epoch 20

`Step2` Train the model on Wikidata15K with forward knowledge transfer from DBpedia15K.
> python train.py --model_path MODEL_PATH1 --kg1 Wikidata15K --kg0 DBpedia15K --embedding_lr 1e-3 --adapter_lr 1e-3 --alpha 1e4 --epoch 40

`Step3` Train the model on Wikidata15K with backward knowledge transfer to improve the performance on DBpedia15K.
> python train.py --model_path MODEL_PATH2 --kg1 Wikidata15K --kg0 DBpedia15K --embedding_lr 1e-3 --adapter_lr 1e-4 --back_transfer --epoch 10

`Step4` Train the model on YAGO15K with forward knowledge transfer from Wikidata15K.
> python train.py --model_path MODEL_PATH2 --kg1 Yago15K --kg0 Wikidata15K --embedding_lr 2e-3 --adapter_lr 1e-3 --alpha 1e4 --epoch 20

`Step5` Train the model on YAGO15K with backward knowledge transfer to improve the performance on Wikidata15K.
> python train.py --model_path MODEL_PATH4 --kg1 Yago15K --kg0 Wikidata15K --embedding_lr 5e-4 --adapter_lr 1e-4 --back_transfer --epoch 10 
