import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import yaml

# Read config
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

test_size = params['data']['test_size']
random_state = params['data']['random_state']

# Load wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Split
train_df, test_df = train_test_split(
    df, test_size=test_size, random_state=random_state
)

# Save
os.makedirs('data', exist_ok=True)
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
print(f'Data prepared: train={len(train_df)}, test={len(test_df)}')