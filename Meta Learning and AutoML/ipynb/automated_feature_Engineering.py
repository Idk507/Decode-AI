import featuretools as ft
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create synthetic dataset
np.random.seed(42)
n_customers = 100
n_orders = 1000

# Customers table
customers = pd.DataFrame({
    'customer_id': range(n_customers),
    'age': np.random.randint(18, 80, n_customers),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers)
})

# Orders table
orders = pd.DataFrame({
    'order_id': range(n_orders),
    'customer_id': np.random.choice(range(n_customers), n_orders),
    'order_date': pd.date_range('2023-01-01', periods=n_orders, freq='H'),
    'amount': np.random.uniform(10, 500, n_orders)
})

# Target: Predict if customer will place a high-value order
labels = pd.DataFrame({
    'customer_id': range(n_customers),
    'high_value': (customers['age'] > 40) & (np.random.rand(n_customers) > 0.3)
})

# Create entity set
es = ft.EntitySet(id='customer_data')
es = es.entity_from_dataframe(entity_id='customers',
                             dataframe=customers,
                             index='customer_id')
es = es.entity_from_dataframe(entity_id='orders',
                             dataframe=orders,
                             index='order_id',
                             time_index='order_date')
es = es.add_relationship(ft.Relationship(es['customers']['customer_id'],
                                       es['orders']['customer_id']))

# Automated feature engineering with Deep Feature Synthesis
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                     target_entity='customers',
                                     agg_primitives=['mean', 'sum', 'count'],
                                     trans_primitives=['month', 'day_of_week'],
                                     max_depth=2)

# Prepare data for modeling
X = feature_matrix.fillna(0)
y = labels['high_value'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Print generated features
print("Generated Features:")
print(feature_matrix.columns)
