import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Step 1: Generate synthetic dataset
data = {
    'Experience': [5, 8, 3, 10, 2, 7],
    'Written_Score': [8, 7, 6, 9, 5, 8],
    'Interview_Score': [10, 6, 7, 8, 9, 5],
    'Salary': [60000, 80000, 45000, 90000, 35000, 75000]
}

df = pd.DataFrame(data)

# Step 2: Save dataset to a .csv file
df.to_csv('candidates_dataset.csv', index=False)

# Step 3: Load dataset
df = pd.read_csv('candidates_dataset.csv')

# Step 4: Split dataset into features and target
X = df.drop('Salary', axis=1)
y = df['Salary']

# Step 5: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize features (optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Build KNN model
knn_model = KNeighborsRegressor(n_neighbors=3)  # Specify the value of K
# Step 8: Train the model
knn_model.fit(X_train_scaled, y_train)

# Step 9: Make predictions on the testing set
y_pred = knn_model.predict(X_test_scaled)

# Step 10: Use the trained model to predict salaries for new candidates
new_candidates = pd.DataFrame({
    'Experience': [5, 8],
    'Written_Score': [8, 7],
    'Interview_Score': [10, 6]
})

# Standardize the new candidate data
new_candidates_scaled = scaler.transform(new_candidates)

# Predict salaries for new candidates
predicted_salaries = knn_model.predict(new_candidates_scaled)
print("Predicted salaries for new candidates:")
for i, salary in enumerate(predicted_salaries):
    print(f"Candidate {i+1}: ${salary:.2f}")
