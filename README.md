# Neural_machine_translation

The goal was to create and train a [sequence-to-sequence](https://developers.google.com/machine-learning/glossary#sequence-to-sequence-task) [Transformer](https://developers.google.com/machine-learning/glossary#Transformer) model to translate [Portuguese into English](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en). The Transformer was originally proposed in ["Attention is all you need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017).
Transformers are deep neural networks that replace CNNs and RNNs with [self-attention](https://developers.google.com/machine-learning/glossary#self-attention). Self attention allows Transformers to easily transmit information across the input sequences.

I followed this tutorial: https://www.tensorflow.org/text/tutorials/transformer  

Steps:  
- Prepare the data.
- Implement necessary components:
  - Positional embeddings.
  - Attention layers.
  - The encoder and decoder.
- Build & train the Transformer.
- Generate translations.
- Export the model.

I used the TensorFlow Datasets to load the Portuguese-English translation dataset Talks Open Translation Project. This dataset contains approximately 52,000 training, 1,200 validation and 1,800 test examples.
 
## Conda environment

```
conda create -n Practice_tf_gpu tensorflow-gpu matplotlib ipykernel pandas seaborn keras-gpu python=3.7
conda activate Practice_tf_gpu
pip install kaggle
pip install transformers
pip install tensorflow_datasets
pip install tensorflow-text
```

## Code
Jupyter notebook with the different steps: Translation_transformer.ipynb  
Python script to run model from terminal: Translation_transformer.py  

## Translate Portuguese into English using final model

```
python ./Translate_port_eng.py "este é o primeiro livro que eu fiz."
```