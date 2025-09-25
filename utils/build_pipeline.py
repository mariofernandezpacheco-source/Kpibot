# training/build_pipeline.py
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBClassifier

from utils.FeatureBuilder import FeatureBuilder
from utils.schemas import OHLCVSchema


# 1) Carga de datos OHLCV + contexto (ya alineados) para ENTRENAMIENTO
def load_train_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def build_and_fit_pipeline(train_df: pd.DataFrame, target_col: str = "label"):
    # Separar X (OHLCV+contexto) e y (label triple-barrera)

    y = train_df[target_col].astype(int)
    X_raw = train_df.drop(columns=[target_col])
    X_raw = OHLCVSchema.validate(X_raw)

    # 2) Feature Builder (misma config_ para live)
    fb = FeatureBuilder(
        price_col_map={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        indicators_cfg={"sma": [20, 50], "ema": [12, 26], "rsi": [14], "atr": [14]},
        lags=3,
        drop_na=False,
    )

    # Para ajustar ColumnTransformer necesitamos conocer columnas numéricas finales
    X_tmp = fb.fit_transform(X_raw)
    numeric_cols = list(X_tmp.columns)  # todo es numérico aquí

    # 3) Preprocesado determinista y robusto
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # winsorización robusta mediante cuantiles (opcional):
            ("quantile", QuantileTransformer(output_distribution="normal", random_state=42)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_cols)], remainder="drop", n_jobs=None
    )

    # 4) Modelo (compatible sklearn)
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )

    pipe = Pipeline(steps=[("features", fb), ("pre", pre), ("clf", clf)])

    pipe.fit(X_raw, y)
    return pipe


def save_pipeline(pipe, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)


if __name__ == "__main__":
    CSV = "01_data/AAPL_5mins_with_labels.csv"  # ajusta a tu ruta real
    OUT = "02_models/AAPL/pipeline.pkl"
    df = load_train_data(CSV)
    pipe = build_and_fit_pipeline(df, target_col="label")
    save_pipeline(pipe, OUT)
    print(f"✅ Guardado: {OUT}")
