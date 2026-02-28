# Scenario: Employee Segmentation in a Tech Company 💻
# Business Problem
# A tech company wants to understand its employees better to design training programs
# and salary structures. They collected data on each employee’s Age and Annual Salary.
# Management believes employees can be grouped into clusters such as:
# - Young, entry‑level employees
# - Mid‑career professionals
# - Senior, high‑earning employees
# They decide to use hierarchical clustering to explore these segments.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Employee Data: [Age, Annual Salary]
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


# Step 1: Scale the data

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


# Step 2: Hierarchical Clustering

hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(data_scaled)

print("Cluster Labels:", labels)

# Step 3: Plot Dendrogram

linked = linkage(data_scaled, method='ward')

plt.figure(figsize=(8,4))
dendrogram(linked,
           orientation='top',
           labels=range(1, len(data)+1))

plt.title('Dendrogram - Employee Segmentation')
plt.xlabel('Employee Index')
plt.ylabel('Distance')
plt.show()