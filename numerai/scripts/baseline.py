"""
numerai/scripts/baseline.py
----------------------------
Baseline LightGBM model for Numerai Classic.
Ported from the official hello_numerai.ipynb into a reusable script.

Goal of this script: establish a known-good CORR/MMC floor before
building anything custom (AE, CST, TFT, ensembles).

Usage:
    python baseline.py --feature-set medium --download

Produces:
    - A trained LightGBM model
    - Validation CORR / MMC / Sharpe / max drawdown
    - A cloudpickled predict() function ready for Model Upload
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import lightgbm as lgb
import cloudpickle
from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr, correlation_contribution

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

napi = NumerAPI()


def get_latest_version() -> str:
    """Numerai iterates dataset versions frequently — always resolve latest."""
    all_datasets = napi.list_datasets()
    versions = sorted(set(d.split("/")[0] for d in all_datasets
                          if d.split("/")[0].startswith("v")))
    return versions[-1]


def download_data(version: str, feature_set_name: str):
    """Download features.json, train, validation, live parquet files."""
    print(f"Downloading {version} datasets...")
    napi.download_dataset(f"{version}/features.json",
                          str(DATA_DIR / f"{version}_features.json"))
    napi.download_dataset(f"{version}/train.parquet",
                          str(DATA_DIR / f"{version}_train.parquet"))
    napi.download_dataset(f"{version}/validation.parquet",
                          str(DATA_DIR / f"{version}_validation.parquet"))
    napi.download_dataset(f"{version}/live.parquet",
                          str(DATA_DIR / f"{version}_live.parquet"))
    print("Download complete.")


def load_feature_set(version: str, feature_set_name: str) -> list:
    # Change f"{version}_features.json" to "features.json"
    meta_path = DATA_DIR / "features.json" 
    metadata = json.load(open(meta_path))
    feature_set = metadata["feature_sets"][feature_set_name]
    print(f"Feature set '{feature_set_name}': {len(feature_set)} features")
    return feature_set


def load_train(version: str, feature_set: list,
              era_subsample: int = 1) -> pd.DataFrame:
    """
    era_subsample: keep every Nth era to reduce memory.
    1 = use all eras (recommended once memory allows).
    """
    # Change f"{version}_train.parquet" to "train.parquet"
    path = DATA_DIR / "train.parquet" 
    train = pd.read_parquet(path, columns=["era", "target"] + feature_set)
    if era_subsample > 1:
        eras = train["era"].unique()[::era_subsample]
        train = train[train["era"].isin(eras)]
    print(f"Train shape: {train.shape}, eras: {train['era'].nunique()}")
    return train


def load_validation(version: str, feature_set: list,
                    era_subsample: int = 1) -> pd.DataFrame:
    path = DATA_DIR / "validation.parquet" 
    val = pd.read_parquet(
        path, columns=["era", "data_type", "target"] + feature_set
    )
    val = val[val["data_type"] == "validation"].drop(columns=["data_type"])
    if era_subsample > 1:
        eras = val["era"].unique()[::era_subsample]
        val = val[val["era"].isin(eras)]
    print(f"Validation shape: {val.shape}, eras: {val['era'].nunique()}")
    return val


def train_baseline(train: pd.DataFrame, feature_set: list,
                   deep: bool = False) -> lgb.LGBMRegressor:
    """
    Two parameter regimes from the official notebook:
    - shallow: fast, good for iteration (~minutes)
    - deep: much better performance, requires substantial CPU/RAM (~hours)
    """
    if deep:
        params = dict(
            n_estimators=30_000, learning_rate=0.001, max_depth=10,
            num_leaves=2**10, colsample_bytree=0.1, min_data_in_leaf=10_000,
        )
    else:
        params = dict(
            n_estimators=2_000, learning_rate=0.01, max_depth=5,
            num_leaves=2**5 - 1, colsample_bytree=0.1,
        )

    print(f"Training LightGBM ({'deep' if deep else 'shallow'} params)...")
    model = lgb.LGBMRegressor(**params)
    model.fit(train[feature_set], train["target"])
    return model


def evaluate(model: lgb.LGBMRegressor, validation: pd.DataFrame,
            feature_set: list, meta_model_version: str = "v4.3",
            meta_model_round: int = 842) -> dict:
    """
    Score against Numerai's own scoring functions.
    CORR: numerai_corr — Numerai-specific Pearson variant, per era
    MMC: correlation_contribution — uniqueness vs the real Meta Model
    """
    validation = validation.copy()
    validation["prediction"] = model.predict(validation[feature_set])

    print(f"Downloading Meta Model ({meta_model_version}, round {meta_model_round})...")
    napi.download_dataset(f"{meta_model_version}/meta_model.parquet",
                          round_num=meta_model_round)
    meta_model = pd.read_parquet(f"{meta_model_version}/meta_model.parquet")
    validation["meta_model"] = meta_model["numerai_meta_model"]

    per_era_corr = validation.groupby("era").apply(
        lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
    )
    per_era_mmc = validation.dropna().groupby("era").apply(
        lambda x: correlation_contribution(
            x[["prediction"]], x["meta_model"], x["target"]
        )
    )

    def summarize(series: pd.Series) -> dict:
        mean = series.mean()
        std = series.std(ddof=0)
        sharpe = mean / std if std > 0 else 0.0
        cum = series.cumsum()
        max_dd = (cum.expanding(min_periods=1).max() - cum).max()
        return dict(mean=mean, std=std, sharpe=sharpe, max_drawdown=max_dd)

    results = {
        "CORR": summarize(per_era_corr["prediction"]),
        "MMC": summarize(per_era_mmc["prediction"]),
    }
    return results, per_era_corr, per_era_mmc


def make_predict_fn(model: lgb.LGBMRegressor, feature_set: list):
    """
    Build the deployable predict function matching Numerai's
    Model Upload interface exactly.
    """
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        live_predictions = model.predict(live_features[feature_set])
        submission = pd.Series(live_predictions, index=live_features.index)
        return submission.to_frame("prediction")
    return predict


def save_deployable(model: lgb.LGBMRegressor, feature_set: list,
                    name: str = "baseline"):
    predict_fn = make_predict_fn(model, feature_set)
    out_path = MODEL_DIR / f"{name}.pkl"
    with open(out_path, "wb") as f:
        cloudpickle.dump(predict_fn, f)
    print(f"Saved deployable model to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default="medium",
                        choices=["small", "medium", "quantum", "all"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--deep", action="store_true",
                        help="Use deep params (slow, better perf)")
    parser.add_argument("--era-subsample", type=int, default=1,
                        help="Keep every Nth era (1 = use all)")
    args = parser.parse_args()

    version = get_latest_version()
    print(f"Using dataset version: {version}")

    if args.download:
        download_data(version, args.feature_set)

    feature_set = load_feature_set(version, args.feature_set)
    train = load_train(version, feature_set, args.era_subsample)
    validation = load_validation(version, feature_set, args.era_subsample)

    model = train_baseline(train, feature_set, deep=args.deep)

    results, per_era_corr, per_era_mmc = evaluate(model, validation, feature_set)

    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    for metric, stats in results.items():
        print(f"\n{metric}:")
        for k, v in stats.items():
            print(f"  {k:>12}: {v:.6f}")

    save_deployable(model, feature_set, name=f"baseline_{args.feature_set}")


if __name__ == "__main__":
    main()