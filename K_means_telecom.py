import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# 1. Create synthetic telecom customer dataset
# --------------------------------------------------
np.random.seed(42)

data = {
    'CustomerID': range(1, 501),
    'MonthlyBill': np.random.randint(20, 200, 500),
    'CallDuration': np.random.randint(50, 500, 500),
    'InternetUsage': np.random.randint(10, 300, 500),
    'SupportCalls': np.random.randint(0, 10, 500)
}

df = pd.DataFrame(data)

# --------------------------------------------------
# 2. Select features (DO NOT use CustomerID)
# --------------------------------------------------
X = df[['MonthlyBill', 'CallDuration', 'InternetUsage', 'SupportCalls']]

# --------------------------------------------------
# 3. Feature Scaling (VERY IMPORTANT for K-Means)
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 4. Elbow Method (WCSS / Inertia)
# --------------------------------------------------
inertia = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method – Telecom Customers')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.show()

# --------------------------------------------------
# 5. Silhouette Score
# --------------------------------------------------
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K = {k}, Silhouette Score = {score:.3f}")

# --------------------------------------------------
# 6. Best K based on Silhouette Score
# --------------------------------------------------
best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\nBest K based on Silhouette Score: {best_k}")

# --------------------------------------------------
# 7. Train Final K-Means Model
# --------------------------------------------------
final_kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster'] = final_kmeans.fit_predict(X_scaled)

print("\nCluster counts:")
print(df['Cluster'].value_counts())

# --------------------------------------------------
# 8. (Optional) Visualize Silhouette Scores
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(K_range, silhouette_scores, marker='o', color='green')
plt.title('Silhouette Score vs K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()