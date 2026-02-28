import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Dataset: Age & Income
data = np.array([
    [25, 15000],
    [28, 16000],
    [30, 18000],
    [35, 22000],
    [40, 25000],
    [45, 60000],
    [50, 65000],
    [55, 70000]
])


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


hierarchical_cluster = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

cluster_labels = hierarchical_cluster.fit_predict(data_scaled)

print("Cluster Labels:", cluster_labels)


linked = linkage(data_scaled, method='ward')

plt.figure(figsize=(8, 4))
dendrogram(linked,
           orientation='top',
           labels=range(1, len(data_scaled)+1),
           distance_sort='descending')

plt.title('Dendrogram - Retail Bank Customers')
plt.xlabel('Customer Index')
plt.ylabel('Distance')
plt.show()