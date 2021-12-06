import pandas
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import operator

data = pandas.read_csv("sentiment_data/baseline_data.csv", header=0, engine='python')
data = data.astype({'commit_id': 'int'})
data['time'] = pandas.to_datetime(data['time'])
data = data.set_index('time')

morning = data.between_time('06:00:00', '11:59:59')
afternoon = data.between_time('12:00:00', '17:59:59')
evening = data.between_time('18:00:00', '22:59:59')
night = data.between_time('23:00:00', '05:59:59')

print("Number of morning comments:", len(morning))
print("Number of afternoon comments:", len(afternoon))
print("Number of evening comments:", len(evening))
print("Number of night comments:", len(night))

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

morning_pos = 0
morning_neg = 0
morning_neu = 0
# print(type(morning))
for row in morning.iterrows():
    # print(row[1])
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
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# sentiment = ['Negative', 'Neutral', 'Positive']
# counts = [morning_neg, morning_neu, morning_pos]
# ax.bar(sentiment,counts)
# plt.show()
    
afternoon_pos = 0
afternoon_neg = 0
afternoon_neu = 0
# print(type(morning))
for row in afternoon.iterrows():
    # print(row[1])
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