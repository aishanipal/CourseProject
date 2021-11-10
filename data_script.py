from getpass import getpass
from mysql.connector import connect, Error

connection = connect(
        host="localhost",
        user="root",
        password=getpass("Password for root: "),
    )
print("Connection at: ", connection)

commit_comments_query = "SELECT user_id, commit_id, body FROM commit_comments;"
f = open("data/user_id_commit_comments.csv", "w+")
cursor = connection.cursor()
cursor.execute("USE mydb")
cursor.execute(commit_comments_query)
f.write("user_id, commit_id, comment\n")
for user_id, commit_id, comment in cursor:
    f.write(str(user_id) + ", " + str(commit_id) + ", " + comment + "\n")
f.close()

commit_comments_query = "SELECT repo_id, user_id FROM project_members;"
f = open("data/repo_id_user_id.csv", "w+")
cursor = connection.cursor()
cursor.execute("USE mydb")
cursor.execute(commit_comments_query)
f.write("repo_id, user_id\n")
for repo_id, user_id in cursor:
    f.write(str(repo_id) + ", " + str(user_id) + "\n")
f.close()