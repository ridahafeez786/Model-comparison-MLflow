import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import pickle

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mlflow.set_tracking_uri('sqlite:///mlflow.db')

# Load params
with open(os.path.join(repo_root, 'params.yaml'), 'r') as f:
    params = yaml.safe_load(f)

model_type = params['model']['type']
model_params = params['model']['params']
test_size = params['data']['test_size']

# Load test data
test_df = pd.read_csv(os.path.join(repo_root, 'data', 'test.csv'))
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# Load trained model
with open(os.path.join(repo_root, 'models', 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Save metrics as JSON for DVC tracking
os.makedirs(os.path.join(repo_root, 'metrics'), exist_ok=True)
metrics_dict = {
    'accuracy': round(acc, 4),
    'f1_score': round(f1, 4),
    'precision': round(precision, 4),
    'recall': round(recall, 4)
}
with open(os.path.join(repo_root, 'metrics', 'scores.json'), 'w') as f:
    json.dump(metrics_dict, f, indent=2)

# Log everything to MLflow
mlflow.set_experiment('wine-classification')
with mlflow.start_run(run_name=f'{model_type}_split{test_size}'):
    mlflow.log_param('model_type', model_type)
    mlflow.log_param('test_size', test_size)
    for k, v in model_params.items():
        mlflow.log_param(k, v)
    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_artifact(os.path.join(repo_root, 'metrics', 'scores.json'))
    print(f'Model: {model_type}')
    print(f'Split: test_size={test_size}')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'MLflow Run ID: {mlflow.active_run().info.run_id}')