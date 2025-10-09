# --- boiling_point_prediction_nordkit.py ---
# Predict boiling points using precomputed descriptors (no RDKit required)

import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Expanded Dataset ---
DATA_STRING = """
SMILES,BoilingPoint,MolWt,HeavyAtomCount,NumHDonors,NumHAcceptors,MolLogP,NumRotatableBonds,RingCount,TPSA,FractionCSP3
C,-161.5,16.04,1,0,0,0.2,0,0,0,1
CC,-88.6,30.07,2,0,0,0.5,0,0,0,1
CCC,-42.1,44.10,3,0,0,1.0,1,0,0,1
CCCC,-0.5,58.12,4,0,0,1.7,2,0,0,1
CCCCC,36.1,72.15,5,0,0,2.3,3,0,0,1
CCCCCC,68.7,86.18,6,0,0,3.0,4,0,0,1
CCCCCCC,98.4,100.20,7,0,0,3.7,5,0,0,1
CCO,78.3,46.07,3,1,1,0.5,1,0,20,0.5
CCCO,97.2,60.10,4,1,1,0.8,2,0,20,0.6
CCCCO,117.7,74.13,5,1,1,1.2,3,0,20,0.7
CC(C)O,82.5,60.10,4,1,1,0.6,1,0,20,0.7
C(C)(C)O,82.3,60.10,4,1,1,0.6,0,0,20,1
c1ccccc1,80.1,78.11,6,0,0,2.3,0,1,0,0
c1ccccc1C,110.6,92.14,7,0,0,2.8,1,1,0,0.2
c1ccccc1CC,136.2,106.17,8,0,0,3.2,2,1,0,0.3
CCCl,12.3,64.51,3,0,0,1.5,1,0,0,0.7
CCCCl,46.6,78.54,4,0,0,2.0,2,0,0,0.8
CCCCCl,78.4,92.57,5,0,0,2.5,3,0,0,0.9
CCBr,38.4,108.97,3,0,0,2.3,1,0,0,0.8
CCCBr,71.0,122.99,4,0,0,2.8,2,0,0,0.9
CCN,16.6,45.08,3,1,1,0.0,1,0,25,0.6
CCCN,47.8,59.11,4,1,1,0.3,2,0,25,0.7
CNC=O,193.0,59.07,3,1,2,-0.2,1,0,40,0.5
CN(C)C=O,153.0,73.10,4,1,2,-0.1,1,0,40,0.6
CC(C)=O,56.3,58.08,4,0,1,0.5,1,0,17,0.8
CCC(=O)C,79.6,72.11,5,0,1,1.1,2,0,17,0.9
CC(=O)O,118.1,60.05,3,1,2,-0.5,1,0,40,0.5
CCC(=O)O,141.0,74.08,4,1,2,0.0,2,0,40,0.6
CCCC(=O)O,163.5,88.11,5,1,2,0.4,3,0,40,0.7
C1CCCCC1,80.7,84.16,6,0,0,2.0,0,1,0,0.0
C1CCOC1,66.0,74.08,5,0,1,0.3,0,1,15,0.0
c1cnccc1,115.2,79.10,6,1,1,0.7,0,1,25,0.0
c1cc[nH]c1,130.0,67.09,5,1,1,0.6,0,1,30,0.0
c1ccco1,31.4,72.07,5,0,1,0.2,0,1,15,0.0
c1ccsc1,84.1,84.13,5,0,0,1.1,0,1,0,0.0
CC#N,81.6,41.05,3,0,1,-0.3,1,0,15,0.6
CCC#N,118.0,55.08,4,0,1,0.0,2,0,15,0.7
C=CC=C,-4.5,54.09,4,0,0,1.8,2,0,0,1.0
C=C(C)C,-6.9,56.11,4,0,0,1.7,1,0,0,1.0
CC(C)=C(C)C,73.0,70.13,5,0,0,2.2,2,0,0,1.0
"""

# --- 2. Load and Train ---
df = pd.read_csv(io.StringIO(DATA_STRING))

descriptor_names = [
    'MolWt','HeavyAtomCount','NumHDonors','NumHAcceptors',
    'MolLogP','NumRotatableBonds','RingCount','TPSA','FractionCSP3'
]

X = df[descriptor_names]
y = df['BoilingPoint']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    results[name] = {
        'R2': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

print("\nModel Performance Summary:")
for name, metrics in results.items():
    print(f"{name:20s} | R² = {metrics['R2']:.3f} | RMSE = {metrics['RMSE']:.2f}")

# --- 3. Visualization ---
best = max(results, key=lambda n: results[n]['R2'])
print(f"\nBest model: {best}")

best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', models[best])
])
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

plt.figure(figsize=(7,7))
sns.scatterplot(x=y_test, y=y_pred, s=70, color="navy")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Boiling Point (°C)")
plt.ylabel("Predicted Boiling Point (°C)")
plt.title(f"Actual vs Predicted Boiling Points ({best})")
plt.grid(True)
plt.tight_layout()
plt.show()
