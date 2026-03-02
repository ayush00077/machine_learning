# Scenario: Online Food Delivery App
# An online food delivery company wants to build a machine learning model to predict delivery time based
#  on the type of cuisine ordered.
# They collect data such as:
# - Cuisine Type: Italian, Chinese, Indian, Mexican
# Since machine learning models can’t directly work with text labels, the company decides to use One-Hot
#  Encoding.
# This method creates a new column for each cuisine type:
# - Italian → [1, 0, 0, 0]
# - Chinese → [0, 1, 0, 0]
# - Indian → [0, 0, 1, 0]
# - Mexican → [0, 0, 0, 1]
# They set sparse=False so the encoder returns a regular NumPy array instead of a sparse matrix, making
#  the results easier to read and print for analysis.

# Questions for Learners
# Part A: Why is One-Hot Encoding more appropriate than Label Encoding for categories like cuisine type?
# Part B: What does sparse=False do, and why might it be useful in this scenario?
# Part C: If the company adds a new cuisine type (e.g., “Thai”), how will One-Hot Encoding handle it?
# Part D (Applied): What potential problem could arise if the company has hundreds of cuisine types, and
#  how might they solve it?


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

orders=pd.DataFrame({
    'cuisine':['italian','chinese','mexican','italian','indian']
})
 
encoder=OneHotEncoder(sparse_output=False)
encoded_array=encoder.fit_transform(orders[['cuisine']])

encoded_df=pd.DataFrame(encoded_array,columns=encoder.categories_[0])

final_data=pd.concat([orders,encoded_df],axis=1)

print("cuisine categories",encoder.categories_)
print('encoded array',encoded_array)
print('final encoded dataframe')
print(final_data)