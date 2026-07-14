import pandas as pd

df = pd.read_csv("C:/Users/dcsuser/Desktop/2022csc016/student_data.csv")
print("Display few rows: ")
print(df.head())

print('\nDescription: ', df.describe())
print('\nNumber of rows and columns: ', df.shape)

# Remove fully repeated rows.
df1 = df.drop_duplicates(inplace=True)
print("\n---Duplicates removed---")
print(df1)

# If the same student appears multiple times, keep only the first entry.
df = df.drop_duplicates(subset=['Student_ID'], keep='first')
print("\n---Duplicates removed---")
print(df)

# Fix Inconsistemt Formatting
df['Gender'] = df['Gender'].str.lower().str.strip()
gender_map = {'male': 'Male', 'female': 'Female', 'f': 'Female', 'm': 'Male'}
df['Gender'] = df['Gender'].map(gender_map)
print("\n---Formatting standarized---")
print(df)

# Fix Incorrect Data Types
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
print("\n---Correct Data Types---")
print(df)

# Conver to numeric and coerce - if conversion fails, force to mark the value as missing (NaN).
import numpy as np
df['Department'] = df['Department'].replace('0', np.nan) # replace numeric value 0 as string 'nan' becuase fill missing value need that column data with same data type
print("\n---Data Type corrected---")
print(df)

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median()) # use 'median' becuase there is an out range value
# if need to fill missing value with number --->  df['Age'] = df['Age'].fillna(30)
df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean()) 
df['Department'] = df['Department'].fillna(df['Department'].mode()[0]) # for catagorical data handling use 'mode'
# fillna - "Fill missing values." It replace NaN with another value.
# mode()[0] - first mode value
print("\n---Missing values handling---")
print(df)

# Handle Noisy Data
df['Department'] = df['Department'].replace('Computer Since', 'Computer Science')
print("\n---Noisy data cleaned---")
print(df)

# Handliing outlier in Salary
# Define the salary limit
upper_limit = 60000
lower_limit = 30000

# Cap salary values
df['Salary'] = np.where(
    df['Salary'] > upper_limit,
    upper_limit,
    np.where(
        df['Salary'] < lower_limit,
        lower_limit,
        df['Salary']
    )
)
print("\n---Outlier salary corrected---")
print(df)

# Handliing outlier in Age
# Define the Age limit
max_age = 22
min_age = 20

# Cap Age values
df['Age'] = np.where(
    df['Age'] > max_age,
    max_age,
    np.where(
        df['Age'] < min_age,
        min_age,
        df['Age']
    )
)
print("\n---Outlier Age corrected---")
print(df)

# Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_minmax = MinMaxScaler()
df['Attendance_Normalixzed'] = scaler_minmax.fit_transform(df[['Attendance']])
print("\n---Min Max Scaler---")
print(df)

# Standardization (Z-score) on Salary (Mean=0, Std=1)
scaler_std = StandardScaler()
df['Salary_Standardized'] = scaler_std.fit_transform(df[['Salary']])
print("\n---Standard Scaler---")
print(df)

# Encoding categorical data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Lable Encoding
le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])
print("\n---Label Encoded---")
print(df)

# One-Hot Encoding for Norminal Data (Gender & City)
df = pd.get_dummies(df, columns=['Gender'], dtype=int)
print("\n---After Categorical Encoding---")
print(df.info())

# Save Cleaned Dataset
df.to_csv('C:/Users/dcsuser/Desktop/2022csc016/student_data_cleaned.csv', index=False)
# index = False - "Do not save the extra index column."
print("\n Cleaned dataset saved as 'Student_data_cleaned.csv'\n")