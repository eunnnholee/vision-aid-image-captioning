import pandas as pd
import re

# Load the captions
captions = pd.read_csv('/Users/eunholee/Downloads/deeplearning_term/flickr8k/captions.txt')
captions['image'] = captions['image'].apply(
    lambda x: f'/Users/eunholee/Downloads/deeplearning_term/flickr8k/images/{x}')
captions.head()

# Preprocess the text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = '[start] ' + text + ' [end]'
    return text
