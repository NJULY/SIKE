# Beyond Re-training: Transferable Representation Learning for Source-Incremental Knowledge Graphs

> Over the years, massive knowledge graphs (KGs) have been constructed one after another, forming continuously expanding sources of structured knowledge. KG representation learning, aimed at embedding entities and relations into a low-dimensional vector space, has shown competitive performance in many knowledge-driven applications. Existing models primarily learn representations on a single KG, ignoring the source-incremental reality, which necessitates efficient representation learning and effective knowledge transfer across KGs. In this paper, we investigate the representation learning of source-incremental KGs and propose a novel model. We design an incremental learning pipeline using a frozen pre-trained language model and KG-specific adapters to learn knowledge and avoid catastrophic forgetting in a sequence of multi-source KGs. To take advantage of the complementary facts of multi-source KGs, we also propose a forward knowledge transfer method to transfer knowledge among KGs and a cross-modal distillation method to distill structural knowledge into text-based representations. To simulate the real-world scenario, we choose three widely used KGs to create a new dataset for evaluating source-incremental KG embedding. The experimental results show that our model can continually learn representations for emerging KGs and benefit from the transferable knowledge in previous KGs and embeddings.

## Dependencies:
- pytorch>=1.10
- transformers>=4.10

## Running
1. Train Struct-Former, and save this model as STRUCT_MODEL
> python former_train.py --tokenizer_path BERT_PATH --data_path DATA_PATH --dataset DATASET --support_dataset SUPPORT_DATASET --support_model_path SUPPORT_MODEL_PATH

where BERT_PATH denotes the path saved the pre-trained language model BERT and its tokenizer, DATA_PATH denotes the root path for DWY15K like xxx/DWY15K, DATASET $\in$ {DBpedia15K, Wikidata15K, Yago15K}, SUPPORT_DATASET $\in$ {DBpedia15K, Wikidata15K, Yago15K, none} in which 'none' denotes DATASET is the first dataset in the sequence, SUPPORT_MODEL_PATH denotes the StructFormer model trained on the last dataset (and 'none' denotes there is no knowledge tranfer).

2. Train Adapter-BERT
> python bert_train.py --bert_path BERT_PATH --model_path MODEL_PATH --struct_model_path STRUCT_MODEL --data_path DATA_PATH  --dataset DATASET --support_dataset SUPPORT_DATASET 

where BERT_PATH denotes the path saved the pre-trained language model BERT and its tokenizer, MODEL_PATH denotes the Adapter-BERT model trained on the last dataset ('none' means to create a new model), STRUCT_MODEL denotes the StructFormer model trained on the current dataset, DATA_PATH denotes the root path for DWY15K like xxx/DWY15K, DATASET $\in$ {DBpedia15K, Wikidata15K, Yago15K}, SUPPORT_DATASET $\in$ {DBpedia15K, Wikidata15K, Yago15K, none} in which 'none' denotes DATASET is the first dataset in the sequence.