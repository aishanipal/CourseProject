import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv("sentiment_data/project289_user6949_scores.txt", delimiter="; ", engine='python')
data = data.dropna()
data = data.astype({'score': 'int'})
data['time'] = pandas.to_datetime(data['time'])

data = data.sort_values('time', ascending=True)
plt.scatter(data['time'], data['score'])
plt.show()