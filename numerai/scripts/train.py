"""
numerai/scripts/train.py
-------------------------
Numerai Classic training pipeline.

Key differences from Kaggle competitions:
- Data is era-based (weekly), not tick/second level
- Metric is CORR (Spearman correlation with target), not RMSE or MAE
- Neutralization is expected — raw predictions are penalized for factor exposure
- Submissions are live: model quality compounds over time on the leaderboard
"""

# TODO: implement after completing all three Kaggle competitions
