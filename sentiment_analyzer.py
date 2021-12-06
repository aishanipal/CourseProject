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
import matplotlib.pyplot as plt

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

data = data.set_index('time')

morning = data.between_time('06:00:00', '11:59:59')
afternoon = data.between_time('12:00:00', '17:59:59')
evening = data.between_time('18:00:00', '22:59:59')
night = data.between_time('23:00:00', '05:59:59')

print("Number of morning comments:", len(morning))
print("Number of afternoon comments:", len(afternoon))
print("Number of evening comments:", len(evening))
print("Number of night comments:", len(night))

morning_pos = 0
morning_neg = 0
morning_neu = 0
for row in morning.iterrows():
    comment = row[1]['clean_comment']
    sentence =  ' '.join(word for word in comment)
    score = sid.polarity_scores(sentence)
    score.pop('compound')
    binary_score = max(score.items(), key=operator.itemgetter(1))[0]
    if binary_score == "neg":
        morning_neg += 1
    elif binary_score == "pos":
        morning_pos += 1
    else:
        morning_neu += 1
print("morning:", morning_neg, morning_neu, morning_pos)

fig = plt.figure()
sentiment = ['Negative', 'Neutral', 'Positive']
counts = [morning_neg, morning_neu, morning_pos]
plt.bar(sentiment,counts)
plt.title("Sentiment of Morning Comments")
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.savefig('plots/morning_baseline.png')

afternoon_pos = 0
afternoon_neg = 0
afternoon_neu = 0
for row in afternoon.iterrows():
    comment = row[1]['clean_comment']
    sentence =  ' '.join(word for word in comment)
    score = sid.polarity_scores(sentence)
    score.pop('compound')
    binary_score = max(score.items(), key=operator.itemgetter(1))[0]
    if binary_score == "neg":
        afternoon_neg += 1
    elif binary_score == "pos":
        afternoon_pos += 1
    else:
        afternoon_neu += 1
print("afternoon:", afternoon_neg, afternoon_neu, afternoon_pos)

fig = plt.figure()
sentiment = ['Negative', 'Neutral', 'Positive']
counts = [afternoon_neg, afternoon_neu, afternoon_pos]
plt.bar(sentiment,counts)
plt.title("Sentiment of Afternoon Comments")
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.savefig('plots/afternoon_baseline.png')

evening_pos = 0
evening_neg = 0
evening_neu = 0
for row in evening.iterrows():
    comment = row[1]['clean_comment']
    sentence =  ' '.join(word for word in comment)
    score = sid.polarity_scores(sentence)
    score.pop('compound')
    binary_score = max(score.items(), key=operator.itemgetter(1))[0]
    if binary_score == "neg":
        evening_neg += 1
    elif binary_score == "pos":
        evening_pos += 1
    else:
        evening_neu += 1
print("evening:", evening_neg, evening_neu, evening_pos)

fig = plt.figure()
sentiment = ['Negative', 'Neutral', 'Positive']
counts = [evening_neg, evening_neu, evening_pos]
plt.bar(sentiment,counts)
plt.title("Sentiment of Evening Comments")
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.savefig('plots/evening_baseline.png')

night_pos = 0
night_neg = 0
night_neu = 0
for row in night.iterrows():
    comment = row[1]['clean_comment']
    sentence =  ' '.join(word for word in comment)
    score = sid.polarity_scores(sentence)
    score.pop('compound')
    binary_score = max(score.items(), key=operator.itemgetter(1))[0]
    if binary_score == "neg":
        night_neg += 1
    elif binary_score == "pos":
        night_pos += 1
    else:
        night_neu += 1
print("night:", night_neg, night_neu, night_pos)

fig = plt.figure()
sentiment = ['Negative', 'Neutral', 'Positive']
counts = [night_neg, night_neu, night_pos]
plt.bar(sentiment,counts)
plt.title("Sentiment of Night Comments")
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.savefig('plots/night_baseline.png')