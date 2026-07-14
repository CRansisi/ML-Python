import pandas as pd # working with tables and datasets
import matplotlib.pyplot as plt # creating charts and graphs
import seaborn as sb # colourful and attractive charts

df = pd.read_csv('C:/Users/dcsuser/Desktop/2022csc016/student_data_cleaned_correct.csv')
print("Ceaned Dataset: ")
print(df)

# Univariate Analysis
plt.figure(figsize=(6,5))
# Counts how many males and females are in the dataset.
gender_count = df['Gender'].value_counts()
# Create bar charts
plt.bar(gender_count.index, gender_count.values, color=['skyblue','pink'])
plt.title('Bar Chart -> Count of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(8,6))
# Create scatter plot
plt.scatter(df['Age'], df['GPA'], color='purple')
plt.title('Scatter Plot -> Age vs GPA')
plt.show()

# Multivariate Analysis
# Correlation Heatmap
plt.figure(figsize=(8,6))
corr = df[['Age', 'GPA', 'Attendance', 'Salary']].corr()
sb.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Heatmap (Correlation)')
plt.show()