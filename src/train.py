import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

# Load params
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

model_type = params['model']['type']
model_params = params['model']['params']

# Load training data
train_df = pd.read_csv('data/train.csv')
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

# Select model based on config
if model_type == 'logistic_regression':
    model = LogisticRegression(**model_params)
elif model_type == 'decision_tree':
    model = DecisionTreeClassifier(**model_params)
elif model_type == 'random_forest':
    model = RandomForestClassifier(**model_params)
elif model_type == 'svm':
    model = SVC(**model_params)
else:
    raise ValueError(f'Unknown model type: {model_type}')

# Train
model.fit(X_train, y_train)

# Save model locally for DVC tracking
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f'Model trained: {model_type}')
print(f'Training samples: {len(X_train)}')