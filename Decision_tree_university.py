# Scenario: University Admissions Rulebook
# Imagine you’re an admissions officer at a university. Every day, students apply for admission, and you need to decide whether to accept or reject them.
# Instead of guessing, you build a rulebook (that’s your Decision Tree).

# 📋 The Data
# - Each applicant has:
# - High School GPA (how well they performed academically)
# - Entrance Exam Score (their standardized test performance)
# - Extracurriculars (1 = active in clubs/sports, 0 = not active)
# - Past applications are labeled:
# - 1 = Accepted
# - 0 = Rejected
# This past data is like your training experience.

# 👉 Just like the loan officer uses credit score, income, and employment status to decide, here the admissions officer uses GPA, exam scores, and extracurriculars to make decisions.

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd # Import pandas as it's used for DataFrame operations

# Loan Dataset
# Label : 1 = Accept, 0=Reject
df=pd.read_csv("/Users/ayush/Desktop/machine_learning/University Dataset - Sheet1.csv")

# Correctly extract features from the DataFrame
X = df[["HighSchool_GPA","Exam_Score","Extracurriculars"]]
# Correctly extract labels from the DataFrame
y = df["Admission_Label"]

x_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train tree (max_depth controls overfitting)
tree = DecisionTreeClassifier(max_depth=3,criterion='gini')
tree.fit(x_train,y_train)

# Visualize the rules
feature_names=["HighSchool_GPA","Exam_Score","Extracurriculars"]
print(export_text(tree,feature_names=feature_names))

# Evaluate
y_pred=tree.predict(x_test)
print("Accuracy : ",accuracy_score(y_test,y_pred))

# New Applicant - Use more realistic values for the features
applicant=[[3.5,1200,1]] # Example: GPA=3.5, Exam=1200, Extracurriculars=1
decision=tree.predict(applicant)
print("Decision:", "ACCEPTED" if decision[0]==1 else "REJECTED")