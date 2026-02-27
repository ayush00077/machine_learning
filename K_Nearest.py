# Scenario 🎬
# A streaming platform wants to recommend movies to users based on their preferences.
#  Each movie is rated on three aspects:
# - Action Rating (how action‑packed it is)
# - Comedy Rating (how funny it is)
# - Drama Rating (how emotional it is)
# The platform collects data from past users about whether they liked (1) or didn’t like (0) certain movies.

# Question for Students
# Using the K‑Nearest Neighbors (KNN) algorithm:
# - Split the dataset into training and testing sets.
# - Scale the features (important for KNN).
# - Train models with different values of K (e.g., 1, 3, 5). Compare their accuracies.
# - Select the best model and predict whether a new user who prefers [Action=4, Comedy=2, Drama=4] will like the movie.
# - Discuss: How does changing K affect the model’s predictions?

# This scenario makes KNN relatable to recommendation systems like Netflix or Spotify, showing students how algorithms decide what they might enjoy.


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Movie dataset
# Features: [action_rating, comedy_rating, drama_rating]
# Label:    1=Will Like, 0=Won't Like
X = [[5,2,3],[4,1,4],[1,5,2],[2,4,1],[5,1,5],
     [3,5,1],[1,4,3],[5,3,4],[2,1,4],[3,4,2]]
y = [1, 1, 0, 0, 1, 0, 0, 1, 1, 0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Scale features (IMPORTANT for KNN!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# Try different values of K
for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"K={k}  Accuracy={acc:.2f}")


# Best model — predict new user
best_knn = KNeighborsClassifier(n_neighbors=3)
best_knn.fit(X_train, y_train)
new_user = scaler.transform([[4, 2, 4]])
print("Will they like it?", best_knn.predict(new_user)[0])