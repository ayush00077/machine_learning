# K-Means Algo

# Example: Customer Segmentation for a Retail Company 🛍️
# Business Context
# A retail chain wants to understand its customers better. Instead of treating everyone the same,
# they want to group customers into segments (like “budget shoppers,” “loyal premium buyers,” etc.)
#  so they can:
# - Personalize marketing campaigns
# - Recommend products more effectively
# - Improve customer retention

# Dataset (simplified)
# Imagine we have customer data with features like:
# - Annual Income (numeric)
# - Spending Score (numeric, based on purchase behavior)
# - Age (numeric)

# Import: Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading the sample data

data = {
    'CustomerID': [1,2,3,4,5,6],
    'Age': [25,45,35,23,52,40],
    'AnnualIncome': [25000,60000,40000,20000,80000,50000],
    'SpendingScore': [30,70,50,20,90,60]
}

df = pd.DataFrame(data)
df.describe


# Select featrues for clustering

X = df[['Age','AnnualIncome','SpendingScore']]

# scale features (important for enterprise datasets)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply for K_Means

kmeans = KMeans(n_clusters = 3, random_state = 42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Inspect results

print(df)

# Visualize clusters (Income VS Spending)

plt.scatter(df['AnnualIncome'], df['SpendingScore'], c = df['Cluster'], cmap = 'viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customers Segments')
plt.show()

