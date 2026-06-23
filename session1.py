import pandas as pd
df = pd.read_csv('C:/Users/dcsuser/Desktop/2022csc016/student_data.csv')
print("Original Dataset: ")
print(df)

print("Display few rows: ")
print(df.head())

print('About Data: ', df.info())
print('Column Names: ', df.columns)

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