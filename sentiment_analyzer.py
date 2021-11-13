import nltk
import pandas
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag_sents, pos_tag

"""
Step 1. Data Collection
"""
data = pandas.read_csv("data/user_id_commit_comments.csv", header=0, delimiter=";;;", engine='python')
data['comment'].dropna(inplace=True)
# print(data)

"""
Step 2. Tokenize Data
"""

# punkt - pre-trained model to tokenize our data
nltk.download('punkt')
data['token_comment'] = data['comment'].apply(word_tokenize)
data['token_comment'].dropna(inplace=True)
token_comments = data.loc[:,'token_comment']
# print(token_comments)

"""
Step 2. POS Tagging
"""

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# data['pos_comment'] = pos_tag_sents(data['token_comment'])
data['pos_comment'] = data['token_comment'].apply(pos_tag)
data['pos_comment'].dropna(inplace=True)
pos_comment = data.loc[:,'pos_comment']
# print(pos_comment)

"""
Step 2. Remove Noise from Data (i.e. stop words)
"""