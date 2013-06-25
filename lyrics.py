from nltk.tokenize import word_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import matplotlib.pyplot as plt

#read file into words and sentences
with open ("data.txt", "r") as myfile:
	raw = myfile.read()
	words = word_tokenize(raw)
	sentences = raw.splitlines()	

#get unique words
unique = list(set(words))

#vectorize the sentences
vectorized = []
n = len(sentences)
for i in xrange(n):
	vector = []
	for j in xrange(len(unique)):
		if unique[j] in sentences[i]:
			vector.append(1)
		else:
			vector.append(0)
	vectorized.append(vector)

#create cosine similarity matrix
dist = np.zeros(n**2).reshape((n, n))
for i in xrange(n):
	for j in xrange(i):
		dist[i][j] = cosine_distance(np.asarray(vectorized[i]), np.asarray(vectorized[j]))
		dist[j][i] = dist[i][j]

#plot it
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dist, interpolation='nearest')
fig.colorbar(cax)
plt.show()
