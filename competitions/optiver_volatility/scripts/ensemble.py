import numpy as np
import sys
from scipy.optimize import minimize
from pathlib import Path

# 1. Dynamically append your scripts directory so we can import train.py
SCRIPTS_DIR = Path('/home/tolugenius/Projects/quant-ml/competitions/optiver_volatility/scripts')
sys.path.append(str(SCRIPTS_DIR))

# Import your exact data loading pipeline to guarantee row/index alignment
from train import load_data

print("Loading aligned targets...")
df = load_data()
y_true = df['target'].values

print("Loading Out-Of-Fold predictions...")
oof_lgb = np.load("oof_lgb.npy")
oof_mlp = np.load("oof_mlp.npy")

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

# Objective function for the optimizer
def objective_func(weight):
    blended_preds = (weight[0] * oof_lgb) + ((1 - weight[0]) * oof_mlp)
    return rmspe(y_true, blended_preds)

print(f"Standalone LightGBM RMSPE: {rmspe(y_true, oof_lgb):.4f}")
print(f"Standalone MLP RMSPE:      {rmspe(y_true, oof_mlp):.4f}")

# Optimize the blend weight
res = minimize(objective_func, [0.5], bounds=[(0.0, 1.0)], method='Nelder-Mead')
best_w = res.x[0]

final_rmspe = objective_func([best_w])

print("\n--- Optimal Ensemble ---")
print(f"LightGBM Weight: {best_w:.4f}")
print(f"MLP Weight:      {1 - best_w:.4f}")
print(f"Blended RMSPE:   {final_rmspe:.4f}")