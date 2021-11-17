""""
@inproceedings{Gousi13,
  author = {Gousios, Georgios},
  title = {The GHTorrent dataset and tool suite},
  booktitle = {Proceedings of the 10th Working Conference on Mining Software
    Repositories},
  series = {MSR '13},
  year = {2013},
  isbn = {978-1-4673-2936-1},
  location = {San Francisco, CA, USA},
  pages = {233--236},
  numpages = {4},
  url = {http://dl.acm.org/citation.cfm?id=2487085.2487132},
  acmid = {2487132},
  publisher = {IEEE Press},
  address = {Piscataway, NJ, USA},
}
"""
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
f.write("user_id;;;commit_id;;;comment\n")
for user_id, commit_id, comment in cursor:
    f.write(str(user_id) + ";;;" + str(commit_id) + ";;;" + comment + "\n")
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

commit_comments_query = "SELECT project_id, commit_id FROM project_commits;"
f = open("data/project_id_commit_id.csv", "w+")
cursor = connection.cursor()
cursor.execute("USE mydb")
cursor.execute(commit_comments_query)
f.write("project_id, commit_id\n")
for project_id, commit_id in cursor:
    f.write(str(project_id) + ", " + str(commit_id) + "\n")
f.close()