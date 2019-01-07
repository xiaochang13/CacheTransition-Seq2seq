# CacheTransition-Seq2seq

This repository contains the code for our paper [Sequence-to-sequence Models for Cache Transition Systems](http://aclweb.org/anthology/P18-1171) in ACL 2018.

The code is developed under TensorFlow 1.0.0, and should be working on later versions of TensorFlow.

Please create issues if there are any questions! This can make things more tractable.

## Data preprocessing

Before training or decoding, there is a required data-preprocessing step, which contains the several tasks introduced below.
To conduct data preprocessing, simply XXXX (TODO of Xiaochang).
As a result, a json file named XXXX (TODO of Xiaochang) will be generated.
In more detail, our data preprocessing conduct the following tasks:

(1) Concept Generation

Concepts are the vertices of target AMR graphs.
Given an input sentence (The boy wants to go), we need to recognize the concepts (boy, want-01, go-01) before generating the corresponding AMR graph.

(2) Anonymization ?? (TODO of Xiaochang, please check whether you apply your EACL work here)

(3) Action Sequence Generation

Based on concepts and the corresponding AMR algorithm, our oracle algorithm determins the action sequences.

## Decoding

Simply execute the corresponding decoding script with one argument being the identifier of the model you want to use. For instance, you can execute "./decode.sh bch20_lr1e3_l21e3"

## Training

First, modify config.json. You should pay attention to the field "suffix", which is an identifier of the model being trained and saved. We usually use the experiment setting, such as "bch20_lr1e3_l21e3", as the identifier. Also point "train_path" and "test_path" to your corresponding files. Note that "test_path" represents the path of a development set, not the final test set.
Finally, execute the corresponding script file, such as "./train.sh".
