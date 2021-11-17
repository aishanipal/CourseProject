"""Tutorial: https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk"""

import nltk
import pandas
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag_sents, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords

"""
Step 1. Data Collection
"""
data = pandas.read_csv("data/user_id_commit_comments.csv", header=0, delimiter=";;;", engine='python')
data = data.dropna()
# print(data)

"""
Step 2. Tokenize Data
"""

# punkt - pre-trained model to tokenize our data
# nltk.download('punkt')
data['token_comment'] = data['comment'].apply(word_tokenize)
data['token_comment'].dropna(inplace=True)
token_comments = data.loc[:,'token_comment']
# print(token_comments)

"""
Step 3. POS Tagging
"""

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# data['pos_comment'] = pos_tag_sents(data['token_comment'])
data['pos_comment'] = data['token_comment'].apply(pos_tag)
data = data.dropna()
pos_comment = data.loc[:,'pos_comment']
# print(pos_comment)

"""
Step 4. Remove Noise from Data (i.e. stop words)
"""

stop_words = stopwords.words('english')

def remove_noise(pos_comment):

    cleaned_tokens = []

    for token, tag in pos_comment:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

data['clean_comment'] = data['pos_comment'].apply(remove_noise)
data = data.dropna()
clean_comment = data.loc[:,'clean_comment']
# print(clean_comment)


"""
Step 5. Determining Word Density

"""

"""
Step 6. Prepare Data for Model

"""

"""
Step 7. Build and Test Model

"""