import pandas
import matplotlib.pyplot as plt

user_ids = [506780, 15246, 9883, 4024, 252614]
graph_type = "line"

for user_id in user_ids:
    user = pandas.read_csv("sentiment_data/project289_user"+str(user_id)+"_scores.txt", delimiter="; ", engine='python')
    user = user.dropna()
    user = user.astype({'score': 'int'})
    user['time'] = pandas.to_datetime(user['time'])
    user = user.sort_values('time', ascending=True)
    if (graph_type == "scatter"):
        plt.scatter(user['time'], user['score'])
    elif (graph_type == "line"):
        plt.plot(user['time'], user['score'])

    plt.xlabel('Time')
    plt.ylabel('Comment Sentiment')
    plt.title('User ' + str(user_id))
    plt.savefig('plots/user'+ str(user_id) + '.png')
    plt.clf()

