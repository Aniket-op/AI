from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define function to evaluate model performance
def evaluate_model(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Vary percentage of training data
percentages = [0.6, 0.7, 0.8, 0.9]
for percentage in percentages:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=percentage, random_state=42)
    accuracy, precision, recall, f1 = evaluate_model(X_train, X_test, y_train, y_test)
    print(f"Percentage of training data: {percentage}")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
    print()

# Explore effect of other decision tree parameters
parameters = {'max_depth': [None, 3, 5, 10], 'min_samples_split': [2, 5, 10]}
for max_depth in parameters['max_depth']:
    for min_samples_split in parameters['min_samples_split']:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_split=min_samples_split)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Max Depth: {max_depth}, Min Samples Split: {min_samples_split}")
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
        print()
