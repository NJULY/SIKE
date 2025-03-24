# Missing Data Recovery for Heterogeneous Graphs with Incremental Multi-Source Data Fusion

## Overview
> Heterogeneous graphs organize data with nodes and edges, and have been widely used in various  graph-centric applications. Often, some data are omitted during manual construction, leading to reduction of data and degeneration of performance on downstream tasks. Existing methods recover the missing data based on the data already within a single graph, neglecting the fact that graphs from different sources share some common nodes due to scopes overlap. In this paper, we concentrate on the missing data recovery task on multi-source heterogeneous graphs under the incremental scenario, designing a novel framework to recover the missing data by fusing multi-source complementary data from previously appeared graphs. Our model, namely SIKE, is present with a pre-trained language model and graph-specific adapters. To take advantage of the complementary data of multi-source graphs, we propose an embedding-based data fusion method to gather data among graphs. To evaluate the proposed model, we choose three widely used heterogeneous graphs and create a new dataset called DWY15K. The experimental results show that our model SIKE achieves significant improvements compared with all competitive baseline models, demonstrating the effectiveness of our model and shedding light on multi-source data fusion for data governance.

## Dependencies:
- python==3.8.12
- pytorch==1.10.0
- adapter-transformers==3.2.1
- numpy==1.23.1
- scipy==1.10.1
- tokenizers==0.10.3
- transformers==4.11.3

## Running DWY15K
The pre-trained language model we used is downloaded at [here](https://huggingface.co/bert-base-cased/tree/main).

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

