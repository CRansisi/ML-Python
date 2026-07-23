import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

wineData = load_wine()

# Features
X = wineData.data

# Labels
y = wineData.target

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model with k = 3
model = KNeighborsClassifier(n_neighbors=9)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Convert predictions to NumPy array for cleaner printing
predictions = np.array(predictions)

print("Predicted classes:")
print(predictions)

print("\nActual classes:")
print(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.2%}")

# k = 1
# Predicted classes:
# [2 0 2 0 1 0 1 2 0 0 2 1 0 1 0 1 1 1 0 1 0 1 0 2 1 2 1 0 1 0 0 1 2 0 0 0]

# Actual classes:
# [0 0 2 0 1 0 1 2 1 2 0 2 0 1 0 1 1 1 0 1 0 1 1 2 2 2 1 1 1 0 0 1 2 0 0 0]

# Accuracy: 77.78%

# k = 3
# Predicted classes:
# [2 0 2 0 1 0 1 2 0 0 2 2 0 1 0 1 1 1 0 1 0 1 2 2 1 2 1 2 1 0 0 1 2 0 0 0]

# Actual classes:
# [0 0 2 0 1 0 1 2 1 2 0 2 0 1 0 1 1 1 0 1 0 1 1 2 2 2 1 1 1 0 0 1 2 0 0 0]

# Accuracy: 80.56%


# k = 5
# Predicted classes:
# [2 0 2 0 1 0 2 0 1 0 2 2 0 1 0 1 1 1 0 1 0 1 2 1 1 1 1 2 1 0 0 1 2 0 0 0]

# Actual classes:
# [0 0 2 0 1 0 1 2 1 2 0 2 0 1 0 1 1 1 0 1 0 1 1 2 2 2 1 1 1 0 0 1 2 0 0 0]

# Accuracy: 72.22%

# k = 7
# Predicted classes:
# [0 0 2 0 1 0 2 0 2 0 2 2 0 1 0 1 1 2 0 1 0 1 2 1 1 1 1 2 1 0 0 1 2 0 0 0]

# Actual classes:
# [0 0 2 0 1 0 1 2 1 2 0 2 0 1 0 1 1 1 0 1 0 1 1 2 2 2 1 1 1 0 0 1 2 0 0 0]

# Accuracy: 69.44%

# k = 9
# Predicted classes:
# [2 0 2 0 1 0 2 2 2 0 2 2 0 1 0 1 1 2 0 1 0 1 2 2 1 2 1 2 1 0 0 1 0 0 0 0]

# Actual classes:
# [0 0 2 0 1 0 1 2 1 2 0 2 0 1 0 1 1 1 0 1 0 1 1 2 2 2 1 1 1 0 0 1 2 0 0 0]

# Accuracy: 72.22%
