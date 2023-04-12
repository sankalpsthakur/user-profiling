import pandas as pd
import numpy as np
import re
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from hmmlearn import hmm
from sklearn.cluster import KMeans


from src.data_preprocessing import (
    process_datetime,
    create_lead_column,
    process_agent_info,
    encode_categorical_variables,
    aggregate_user_level_data,
)

from src.feature_engineering import create_user_features


data = pd.read_csv('data/clickstream_data.csv')

# Data Preprocessing 
data = process_datetime(data)
data = create_lead_column(data)
data = process_agent_info(data)
data = encode_categorical_variables(data)

user_level_data = aggregate_user_level_data(data)

# Feature Engineering

## User Features
user_level_data = create_user_features(user_level_data)

## Polynomial Features
numerical_columns = ['total_events', 'avg_time_per_session'] + list(event_proportions.columns) + list(sessions_per_device.columns)
imputer = SimpleImputer(strategy='mean')
user_level_data_imputed = user_level_data[numerical_columns].copy()
user_level_data_imputed[numerical_columns] = imputer.fit_transform(user_level_data[numerical_columns])
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = pd.DataFrame(poly.fit_transform(user_level_data_imputed), columns=poly.get_feature_names_out(numerical_columns), index=user_level_data.index)
log_normalized_columns = ['total_events', 'avg_time_per_session']
user_level_data[log_normalized_columns] = user_level_data[log_normalized_columns].applymap(lambda x: np.log1p(x))
user_level_data = pd.concat([user_level_data, poly_features], axis=1)

# Split data into train and test sets
X = user_level_data.drop(columns=['lead'])
y = user_level_data['lead']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
numerical_features = list(X.select_dtypes(include=[np.number]).columns)
categorical_features = list(X.select_dtypes(exclude=[np.number]).columns)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='drop')

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define the models and hyperparameter grids
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

param_grids = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
    'Decision Tree': {'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 5, 10]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Train the models
grid_searches = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], scoring='roc_auc', cv=5, n_jobs=-1)
    grid_search.fit(X_train_processed, y_train)
    grid_searches[name] = grid_search

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_processed)

# Consensus model
consensus_pred = np.zeros(y_test.shape)
weights = [0.25, 0.25, 0.25, 0.25]  # Adjust these weights as required (they should sum up to 1)
probs = []

for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], scoring='roc_auc', cv=5, n_jobs=-1)
    grid_search.fit(X_train_processed, y_train)
    best_model = grid_search.best_estimator_
    prob = best_model.predict_proba(X_test_processed)[:, 1]
    probs.append(prob)

# Compute the consensus probability prediction using the weighted average
consensus_prob = np.average(probs, axis=0, weights=weights)

# Compute the final binary predictions using a 0.5 threshold
consensus_pred = (consensus_prob > 0.5).astype(int)

# Evaluate the consensus model
print("Consensus Model Performance (with weighted averaging):")
print(f"Accuracy: {accuracy_score(y_test, consensus_pred):.2f}")
print(f"Precision: {precision_score(y_test, consensus_pred):.2f}")
print(f"Recall: {recall_score(y_test, consensus_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, consensus_pred):.2f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, consensus_prob):.2f}\n")
