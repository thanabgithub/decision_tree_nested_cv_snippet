from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a dummy classifier using the most frequent class as the baseline
dummy = DummyClassifier(strategy='most_frequent')

# Fit the dummy classifier on the training data
dummy.fit(X_train, y_train)

# Make predictions on the testing data using the trained dummy classifier
y_pred = dummy.predict(X_test)

# Calculate the accuracy of the baseline model
baseline_accuracy = accuracy_score(y_test, y_pred)

print(f"Baseline model accuracy: {baseline_accuracy:.2f}")
