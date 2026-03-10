# Description
# Capstone Project: Student Success & Career Path Prediction

# Scenario

# The university wants to analyze student performance data to:

# Predict exam scores (Regression).
# Classify students into “At Risk” vs. “On Track” categories (Classification).
# Cluster students into groups with similar study habits (Clustering).
# Recommend interventions (extra tutoring, workshops, counseling).


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

# Load dataset
df = pd.read_csv('/Users/ayush/Desktop/machine_learning/capstone_projects/Student Success & Career Path  - Sheet1 (1).csv')

print("Columns in dataset:")
print(df.columns)

# Fill missing numeric values
df = df.fillna(df.median(numeric_only=True))

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])


# REGRESSION (Use last numeric column)

print(" REGRESSION ")

numeric_cols = df.select_dtypes(include='number').columns
target_reg = numeric_cols[-1]   # Last numeric column

X_reg = df.drop(target_reg, axis=1)
y_reg = df[target_reg]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

reg_pred = reg_model.predict(X_test)

print("Target used for regression:", target_reg)
print("MAE:", mean_absolute_error(y_test, reg_pred))

# CLASSIFICATION (Use last column)


print("CLASSIFICATION")

target_clf = df.columns[-1]   # Last column

X_clf = df.drop(target_clf, axis=1)
y_clf = df[target_clf]

X_scaled = StandardScaler().fit_transform(X_clf)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)

clf_model = LogisticRegression()
clf_model.fit(X_train, y_train)

clf_pred = clf_model.predict(X_test)

print("Target used for classification:", target_clf)
print("Accuracy:", accuracy_score(y_test, clf_pred))
print(classification_report(y_test, clf_pred))

# CLUSTERING


print("CLUSTERING")

X_cluster = StandardScaler().fit_transform(df)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

print(df[['Cluster']].head())

# Recommendation
def recommend(cluster):
    if cluster == 0:
        return "Extra Tutoring"
    elif cluster == 1:
        return "Workshop"
    else:
        return "Counseling"

df['Recommendation'] = df['Cluster'].apply(recommend)

print("\nSample Recommendations:")
print(df[['Cluster', 'Recommendation']].head())
