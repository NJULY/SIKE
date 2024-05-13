# Transferable Representation Learning for Source-Incremental Knowledge Graphs

## Overview
> Knowledge graph (KG) representation learning, which aims at embedding entities and relations into a low-dimensional vector space, has shown competitive performance in many knowledge-driven applications. With the requirements of applications changing or expanding, KGs from different sources may be introduced one after another over time. Existing work primarily learns representations on a single KG while ignoring the source-incremental reality, which necessitates efficient representation learning and effective knowledge transfer across KGs. In this paper, we investigate a new representation learning scenario of source-incremental KGs and propose a novel model named SIKE. We present an incremental learning pipeline using a frozen pre-trained language model and KG-specific adapters to learn knowledge and avoid catastrophic forgetting in a sequence of multi-source KGs. To take advantage of the transferable knowledge of multi-source KGs, we design a bidirectional knowledge transfer method to transfer knowledge among KGs. To simulate the real-world scenario, we choose three widely-used KGs and create a new dataset for evaluating source-incremental KG representation learning. The experimental results show that our model SIKE continually learns representations for emerging KGs, and achieves significant forward knowledge transfer and positive backward knowledge transfer at the same time, benefiting from the transferable knowledge among KGs.

## Dependencies:
- pytorch>=1.10
- transformers>=4.10

## Running
See the commands in the folder `scripts`.
