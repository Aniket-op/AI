#data set generator
import pandas as pd
import numpy as np

# Generate random data
np.random.seed(42)
data = pd.DataFrame(np.random.rand(100, 5), columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])

# Generate random target class (0 or 1)
data['TARGET CLASS'] = np.random.randint(0, 2, size=len(data))

# Save data to CSV file
data.to_csv('classified_data.csv', index=False)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Classified Data
classified_data = pd.read_csv('classified_data.csv')

# Prepare data
X = classified_data.drop('TARGET CLASS', axis=1)
y = classified_data['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define functions to evaluate models
def evaluate_knn(n_neighbors):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def evaluate_nb():
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def evaluate_decision_tree(max_depth, min_samples_split):
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Vary parameters for each model
knn_parameters = [1, 3, 5, 7, 9]
nb_parameters = []
decision_tree_parameters = [{'max_depth': None, 'min_samples_split': 2},
                            {'max_depth': 5, 'min_samples_split': 2},
                            {'max_depth': 10, 'min_samples_split': 2}]

# Evaluate models and save results to CSV
results = []
for n_neighbors in knn_parameters:
    accuracy, precision, recall, f1 = evaluate_knn(n_neighbors)
    results.append(['KNN', n_neighbors, accuracy, precision, recall, f1])

accuracy, precision, recall, f1 = evaluate_nb()
results.append(['Naive Bayes', 'N/A', accuracy, precision, recall, f1])

for params in decision_tree_parameters:
    accuracy, precision, recall, f1 = evaluate_decision_tree(params['max_depth'], params['min_samples_split'])
    results.append(['Decision Tree', f"Max Depth: {params['max_depth']}, Min Samples Split: {params['min_samples_split']}", accuracy, precision, recall, f1])

# Create DataFrame and save to CSV
df = pd.DataFrame(results, columns=['Model', 'Parameters', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
df.to_csv('model_comparison_results.csv', index=False)
