import pandas as pd

df = pd.read_csv("C:/Users/dcsuser/Desktop/2022csc016/student_data.csv")
print("Display few rows: ")
print(df.head())

print('Description: ', df.describe())
print('Number of rows and columns: ', df.shape)

# Remove fully repeated rows.
df1 = df.drop_duplicates(inplace=True)
print("Duplicates removed.")
print(df1)

# If the same student appears multiple times, keep only the first entry.
df = df.drop_duplicates(subset=['Student_ID'], keep='first')
print("Duplicates removed.")
print(df)

# Fix Inconsistemt Formatting
df['Gender'] = df['Gender'].str.lower().str.strip()
gender_map = {'male': 'Male', 'female': 'Female', 'f': 'Female', 'm': 'Male'}
df['Gender'] = df['Gender'].map(gender_map)
print("Formatting standarized.")
print(df)

# Fix Incorrect Data Types
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
print("Correct Data Types")
print(df)

# Conver to numeric and coerce - if conversion fails, force to mark the value as missing (NaN).
import numpy as np
df['Department'] = df['Department'].replace('0', np.nan) # replace numeric value 0 as string 'nan' becuase fill missing value need that column data with same data type
print("Data Type corrected.")
print(df)

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median()) # use 'median' becuase there is an out range value
# if need to fill missing value with number --->  df['Age'] = df['Age'].fillna(30)
df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean()) 
df['Department'] = df['Department'].fillna(df['Department'].mode()[0]) # for catagorical data handling use 'mode'
# fillna - "Fill missing values." It replace NaN with another value.
# mode()[0] - first mode value
print("Missing values handling")
print(df)

# Handle Noisy Data
df['Department'] = df['Department'].replace('Computer Since', 'Computer Science')
print("Noisy data cleaned.")
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
print("Outlier salary corrected.")
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
print("Outlier Age corrected.")
print(df)

# Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_minmax = MinMaxScaler()
df['Attendance_Normalixzed'] = scaler_minmax.fit_transform(df[['Attendance']])
print("Min Max Scaler.")
print(df)

# Standardization (Z-score) on Salary (Mean=0, Std=1)
scaler_std = StandardScaler()
df['Salary_Standardized'] = scaler_std.fit_transform(df[['Salary']])
print("Standard Scaler.")
print(df)