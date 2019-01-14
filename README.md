# CacheTransition-Seq2seq

This repository contains the code for our paper [Sequence-to-sequence Models for Cache Transition Systems](http://aclweb.org/anthology/P18-1171) in ACL 2018.

The code is developed under TensorFlow 1.0.0, and should be working on later versions of TensorFlow.

Please create issues if there are any questions! This can make things more tractable.

## Data preprocessing

Before training or decoding, there is a required data preprocessing procedure. This includes a series of more specific data preprocessing steps mentioned in the paper:

(1) Running JAMR
This step relies on running JAMR (https://github.com/jflanigan/jamr) on the AMR data to get the pos tag, dependency information. For example: 
${JAMR_HOME}/scripts/train_LDC2015E86.sh

This will generate the results for tokenization, dependency parsing and NER tagging, extract alignment for the train, eval and test split. We also extract the concept identification results from JAMR:
${JAMR_HOME}/scripts/TRAIN_STAGE1_ONLY.sh

(2) Categorization (Anonymization)
We first collect all the JAMR output from the previous step. Then we run the the categorization step to collapse multi-concept categories (named-entity, dates, verbalization et al.). See pipeline.sh for more details (need to change two paths according to the output from the previous step).

(3) Action Sequence Generation
Given the input text, target categorized AMR and the alignment between them, we use the oracle algorithm to compute the oracle action sequence. See pipeline.sh for more details.

## Decoding

Simply execute the corresponding decoding script with one argument being the identifier of the model you want to use. For instance, you can execute "./decode.sh bch20_lr1e3_l21e3"

## Training

First, modify config.json. You should pay attention to the field "suffix", which is an identifier of the model being trained and saved. We usually use the experiment setting, such as "bch20_lr1e3_l21e3", as the identifier. Also point "train_path" and "test_path" to your corresponding files. Note that "test_path" represents the path of a development set, not the final test set.
Finally, execute the corresponding script file, such as "./train.sh".

## Cite

If you like our paper, please cite
```
@InProceedings{peng-acl18,
  author = {Xiaochang Peng and Linfeng Song and Daniel Gildea and Giorgio
  Satta},
  title = {Sequence-to-sequence Models for Cache Transition Systems},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for
  Computational Linguistics (ACL-18)},
  year = {2018},
  pages = {1842--1852},
  URL = {https://www.cs.rochester.edu/u/gildea/pubs/peng-acl18.pdf}
}
```
