import pandas as pd
import tensorflow as tf

# Load the captions
captions = pd.read_csv('/Users/david/Downloads/deeplearning_term/flickr8k/captions.txt')
captions['image'] = captions['image'].apply(
    lambda x: f'/Users/david/Downloads/deeplearning_term/flickr8k/images/{x}')

# Configuration constants
MAX_LENGTH = 40
VOCABULARY_SIZE = 15000

# Initialize the TextVectorization layer
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    standardize=None,
    output_sequence_length=MAX_LENGTH)

# Adapt the tokenizer to the captions
tokenizer.adapt(captions['caption'])

# Create word to index mapping
word2idx = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())

# Create index to word mapping
idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

# Optional: Print the first 5 tokens in the vocabulary
print(tokenizer.get_vocabulary()[:5])
