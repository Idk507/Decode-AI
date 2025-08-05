import autosklearn.classification
import sklearn.datasets
import sklearn.metrics
import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic classification dataset
X, y = sklearn.datasets.make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Auto-sklearn classifier
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,  # Run for 2 minutes
    per_run_time_limit=30,        # Limit each model to 30 seconds
    n_jobs=-1,                    # Use all available CPU cores
    seed=42
)

# Fit the model (performs model selection and hyperparameter tuning)
automl.fit(X_train, y_train)

# Predict and evaluate
y_pred = automl.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Print the selected model(s) and ensemble
print("Selected Models:")
print(automl.show_models())
