import sys
import matplotlib.pyplot as plt
import csv

file = "data/experiments/" + sys.argv[1] + "/metrics.csv"

episodes = []
states = []

with open(file, "r") as csvfile:
    plots = csv.reader(csvfile,delimiter=',')

    counter = 0
    for row in plots:
        episodes.append(row[0])
        states.append(row[6])
        #if(counter % 100 == 0):

        #counter += 1


plt.plot(episodes,states)


plt.xlabel("episodes")
plt.ylabel("states in Q-table")

plt.title(f'experiment {sys.argv[1]}')  

plt.show()










