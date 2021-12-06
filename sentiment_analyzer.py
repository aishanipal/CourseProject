"""Tutorial: https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk"""

import nltk
import pandas
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag_sents, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re, string
from nltk.corpus import stopwords
import json
import operator

"""
Step 1. Data Collection
"""
data = pandas.read_csv("data/user_id_commit_comments.csv", header=0, delimiter=";;;", engine='python')
data = data.dropna()
data = data.astype({'commit_id': 'int'})
data['time'] = pandas.to_datetime(data['time'])
project_commit_ids = pandas.read_csv("data/project_id_commit_id.csv", header=0, delimiter=", ", engine='python')
grouped_project_commit_ids = project_commit_ids.groupby('project_id').agg(pandas.Series.tolist)
# grouped_project_commit_ids.reset_index()
# print(grouped_project_commit_ids.columns)
# for each row in grouped_project_commit_ids: 
    # to get project id - row[0], to get list of commit_id - list(row[1])[0]
# for project_id, commits in grouped_project_commit_ids.iterrows():
#     print("project_id:", project_id)
#     print("num commits:", len(commits[0]))

"""
Step 2. Tokenize Data
"""

# punkt - pre-trained model to tokenize our data
# nltk.download('punkt')
data['token_comment'] = data['comment'].apply(word_tokenize)
data = data.dropna()
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
Step 5. Classify (using pre-trained models: https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c)

"""
#nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
commit_ids = project_commit_ids.loc[project_commit_ids['project_id'] == 289]['commit_id']
project289 = data.loc[data['commit_id'].isin(commit_ids)]
gp = project289.groupby('user_id')
# print(gp.ngroups)

for g in gp.groups:
    f = open("sentiment_data/project289_user" + str(g) + "_scores.txt", "w+", encoding="utf-8")
    f.write("comment; score; time\n")
    user = gp.get_group(g)
    user_commit_ids = user['commit_id']
    # print(user_commit_ids)

    for commit_id in user_commit_ids:
        comments = list(data.loc[data['commit_id'] == commit_id]['clean_comment'])
        # print(commit_id, comments)
        times = list(data.loc[data['commit_id'] == commit_id]['time'])
        assert len(comments) == len(times)
        for i in range(len(comments)):
            comment = comments[i]
            time = times[i]
            sentence =  ' '.join(word for word in comment)
            score = sid.polarity_scores(sentence)
            score.pop('compound')
            binary_score = max(score.items(), key=operator.itemgetter(1))[0]
            bin = 0
            if binary_score == "neg":
                bin = -1
            elif binary_score == "pos":
                bin = 1
            f.write(sentence + "; " + str(bin) + "; " + str(time) + "\n")
        
    f.close()


"""
Prep data for baseline

"""
data.to_csv('sentiment_data/baseline_data.csv')