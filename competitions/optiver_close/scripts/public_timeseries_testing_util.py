"""
competitions/optiver_close/scripts/public_timeseries_testing_util.py
--------------------------------------------------------------------
A local mock of the Kaggle time-series API for the Optiver 2023 competition.
"""
import pandas as pd
from pathlib import Path

class MockEnv:
    def __init__(self):
        print("Initializing Mock API Environment...")
        # Resolve the path to your raw training data
        data_dir = Path(__file__).resolve().parents[3] / "data" / "optiver-close-trading"
        
        # Load just the last few days to keep the simulation fast and avoid memory spikes
        df = pd.read_csv(data_dir / "train.csv")
        self.df = df[df["date_id"] >= 478].reset_index(drop=True)
        
        # Group by exact time step to stream it sequentially
        self.groups = self.df.groupby(["date_id", "seconds_in_bucket"])
        self.keys = list(self.groups.groups.keys())
        
    def iter_test(self):
        """Yields one bucket of data at a time."""
        for key in self.keys:
            test_df = self.groups.get_group(key).copy()
            
            # The API provides a submission template for the current bucket
            sample_prediction_df = test_df[["row_id"]].copy()
            sample_prediction_df["target"] = 0.0
            
            # Revealed targets (empty dummy df for this competition format)
            revealed_targets = pd.DataFrame()
            
            yield test_df, revealed_targets, sample_prediction_df
            
    def predict(self, sample_prediction_df):
        """Simulates receiving your predictions."""
        # In a real Kaggle environment, this saves the predictions to memory.
        # Here we just silently accept them to keep the loop moving instantly.
        pass

def make_env():
    return MockEnv()