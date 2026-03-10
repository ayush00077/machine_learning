#  Scenario: Healthcare Patient Records Encoding
# Imagine you are working as a data scientist in a hospital. The hospital wants to build a machine
#  learning model to predict patient recovery time based on demographic and treatment details.
#   The dataset contains categorical variables (like Treatment Type and Hospital Wing) that must be converted into numeric form before modeling.
# Business Context
# - Treatment Type: Different medical procedures (Surgery, Therapy, Medication).
# - Hospital Wing: Location where the patient is admitted (East, West, North, South).
# - Recovery Days: Numeric values representing how many days the patient took to recover.
# The challenge:
# Machine learning models cannot directly interpret text categories, so you need to encode categorical
#  features into numbers.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame({
    'Treatment_Type': ['surgery', 'therapy', 'edication'],
    'Hospital_Wing': ['North', 'South', 'East'],
    'Recovery_Days': [20, 15, 30]
})

le_treatment = LabelEncoder()
le_wing = LabelEncoder()

# Apply encoding
df["Treatment_Label"] = le_treatment.fit_transform(df["Treatment_Type"])
df["Wing_Label"] = le_wing.fit_transform(df["Hospital_Wing"])

print(df[['Treatment_Type', 'Treatment_Label']])
print(df[['Hospital_Wing', 'Wing_Label']])



encoder = OneHotEncoder(sparse_output=False)

encoded_array = encoder.fit_transform(df[["Treatment_Type", "Hospital_Wing"]])

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(["Treatment_Type", "Hospital_Wing"])
)

# Combine with original dataframe
final_df = pd.concat([df[["Recovery_Days"]], encoded_df], axis=1)

print(final_df)

