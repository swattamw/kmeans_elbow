import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
#GETTING DATA
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data['feature_names'])

#KMEANS LOOPS ELBOW METHOD
sse = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, max_iter=10000).fit(iris)
    iris["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_

#PLOTTING SSE VALUES AGAINST K-VALUES
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xticks(np.arange(min(sse.keys()), max(sse.keys())+1, 1.0))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()

