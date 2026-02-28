#  Scenario: Loan Officer’s Rulebook
# Imagine you’re a loan officer at a bank. Every day, people apply for loans, and you need to decide whether to approve or reject them.
# Instead of guessing, you build a rulebook (that’s your Decision Tree).

# 📋 The Data
# - Each applicant has:
# - Credit Score (how trustworthy they are with money)
# - Income (in thousands)
# - Employment status (1 = employed, 0 = not employed)
# - Past applications are labeled:
# - 1 = Approved
# - 0 = Rejected
# This past data is like your training experience.

 
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loan Dataset
# Features : [Credit_Score, source_income, employed]
# Label : 1 =Approve, 0=Reject
X = [
    [720, 60, 1], [580, 35, 0], [700, 55, 1],
    [600, 40, 1], [750, 80, 1], [500, 25, 0],
    [680, 50, 1], [550, 30, 0], [730, 70, 1],
    [610, 42, 0],
]
y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

x_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train tree (max_depth controls overfitting)
tree = DecisionTreeClassifier(max_depth=3,criterion='gini')
tree.fit(x_train,y_train)

# Visualize the rules
feature_names=['credit_score','income','employed']
print(export_text(tree,feature_names=feature_names))

# Evaluate
y_pred=tree.predict(x_test)
print("Accuracy : ",accuracy_score(y_test,y_pred))

# New Applicant
applicant=[[0,50,0]]
decision=tree.predict(applicant)
print("Decision:", "APPROVED" if decision[0]==1 else "REJECTED")






