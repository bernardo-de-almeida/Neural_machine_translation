
# example: python ./Translate_port_eng.py "este é o primeiro livro que eu fiz."

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import tensorflow as tf
import tensorflow_text

# get input
import sys
sentences = sys.argv[1:]

# Load model
translator = tf.saved_model.load('translator')

def print_translation(sentence, tokens):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Translation":15s}: {translated_text.numpy().decode("utf-8")}')

# Translate
print("\nPredictions ...\n")
for s in sentences:
    translated_text = translator(s)
    print_translation(s, translated_text)
    print()