"""
Neft Emalı Optimallaşdırma Sistemi
Modul: preprocessor.py
Məqsəd: Sensor məlumatının filtrasiyası, normallaşdırılması,
        anomaliya aşkarlanması və xətti reqressiya modelləşdirilməsi.

Reqressiya arxitekturası:
    Hər hədəf dəyişəni üçün LinearRegression öyrədilir.
    Model: Y = a1*T + a2*P + a3*F + ... + a9*S + b
    Metrikalar: R², RMSE, əmsallar (feature coefficients)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

INPUT_FEATURES = [
    "furnace_temp", "column_pressure", "flow_rate",
    "reflux_ratio", "feed_temp", "h2_pressure",
    "catalyst_temp", "crude_density", "sulfur_content",
]

TARGET_COLS = [
    "yield_gasoline", "yield_diesel", "yield_kerosene",
    "energy_gj_h", "sulfur_removal", "total_yield",
]

CV_FOLDS = 3


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sensor sütunlarını interpolasiya edir, 3-sigma klipinqlə həddə gətirir
    və NaN sətirləri silir.
    """
    df = df.copy()
    for col in INPUT_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].interpolate(method="linear", limit=5)
        mu, sigma = df[col].mean(), df[col].std()
        df[col] = df[col].clip(mu - 3 * sigma, mu + 3 * sigma)
    df = df.dropna(subset=INPUT_FEATURES).reset_index(drop=True)
    return df


class DataNormaliser:
    """Min-Maks normallaşdırma — X və y üçün ayrı scaler."""

    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self._fitted  = False

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        X = df[INPUT_FEATURES].values
        y = df[TARGET_COLS].values
        X_norm = self.scaler_X.fit_transform(X)
        y_norm = self.scaler_y.fit_transform(y)
        self._fitted = True
        return (
            pd.DataFrame(X_norm, columns=INPUT_FEATURES),
            pd.DataFrame(y_norm, columns=TARGET_COLS),
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Əvvəlcə fit_transform çağırın."
        return pd.DataFrame(
            self.scaler_X.transform(df[INPUT_FEATURES].values),
            columns=INPUT_FEATURES,
        )

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        return self.scaler_y.inverse_transform(y_norm)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    İki paralel metod:
        anomaly_iqr      — IQR qaydası (hər sensor üçün)
        anomaly_iforest  — IsolationForest (çoxölçülü)
    anomaly_final = anomaly_iqr OR anomaly_iforest
    """
    df = df.copy()

    iqr_flag = pd.Series(False, index=df.index)
    for col in INPUT_FEATURES:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        iqr_flag |= (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    df["anomaly_iqr"] = iqr_flag

    iso    = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    preds  = iso.fit_predict(df[INPUT_FEATURES])
    scores = iso.decision_function(df[INPUT_FEATURES])
    df["anomaly_iforest"] = preds == -1
    df["anomaly_score"]   = scores
    df["anomaly_final"]   = df["anomaly_iqr"] | df["anomaly_iforest"]
    return df


class LinearRegressionModel:
    """
    Hər hədəf dəyişəni üçün ayrı LinearRegression öyrədilir.
    Y = a1*x1 + a2*x2 + ... + a9*x9 + b

    metrics cədvəlinin sütunları:
        CV_R², R², RMSE
    Coefficients: hər feature üçün əmsal (intercept daxil)
    """

    def __init__(self):
        self.models  : dict[str, LinearRegression] = {}
        self.metrics : dict[str, dict]             = {}
        self.coeffs  : dict[str, np.ndarray]       = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LinearRegressionModel":
        X_df  = X[INPUT_FEATURES]
        X_arr = X_df.values

        for target in TARGET_COLS:
            y_t = y[target].values

            model = LinearRegression()

            cv_r2 = cross_val_score(
                model, X_arr, y_t,
                cv=CV_FOLDS, scoring="r2", n_jobs=1,
            ).mean()

            model.fit(X_df, y_t)
            pred = model.predict(X_df)

            self.models[target]  = model
            self.coeffs[target]  = np.append(model.coef_, model.intercept_)

            self.metrics[target] = {
                "CV_R²": round(cv_r2, 4),
                "R²":    round(r2_score(y_t, pred), 4),
                "RMSE":  round(np.sqrt(mean_squared_error(y_t, pred)), 6),
            }

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Əvvəlcə fit çağırın."
        X_df  = X[INPUT_FEATURES]
        preds = {t: self.models[t].predict(X_df) for t in TARGET_COLS}
        return pd.DataFrame(preds, index=X.index)

    def get_metrics_df(self) -> pd.DataFrame:
        """Sütunlar: CV_R², R², RMSE | İndeks: TARGET_COLS"""
        return pd.DataFrame(self.metrics).T

    def get_coefficients_df(self) -> pd.DataFrame:
        """
        Hər hədəf üçün normallaşdırılmış feature əmsalları.
        Sütunlar: INPUT_FEATURES + ['intercept'] | İndeks: TARGET_COLS
        """
        rows = {}
        col_names = INPUT_FEATURES + ["intercept"]
        for target in TARGET_COLS:
            rows[target] = dict(zip(col_names, self.coeffs[target]))
        return pd.DataFrame(rows).T


class AROPipeline:
    """
    Tam pipeline:
        filter_data → DataNormaliser → LinearRegressionModel
    """

    def __init__(self):
        self.normaliser = DataNormaliser()
        self.regressor  = LinearRegressionModel()
        self._ready     = False

    def fit(self, raw_df: pd.DataFrame) -> "AROPipeline":
        df_clean       = filter_data(raw_df)
        X_norm, y_norm = self.normaliser.fit_transform(df_clean)
        self.regressor.fit(X_norm, y_norm)
        self._ready = True
        return self

    def predict_normalised(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        assert self._ready, "Pipeline fit edilməyib."
        return self.regressor.predict(self.normaliser.transform(X_raw))

    def predict_physical(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        y_norm = self.predict_normalised(X_raw).values
        y_phys = self.normaliser.inverse_transform_y(y_norm)
        return pd.DataFrame(y_phys, columns=TARGET_COLS, index=X_raw.index)

    @property
    def metrics(self) -> pd.DataFrame:
        return self.regressor.get_metrics_df()

    @property
    def coefficients(self) -> pd.DataFrame:
        return self.regressor.get_coefficients_df()


def run_preprocessing_pipeline(raw_df: pd.DataFrame) -> dict:
    """
    Qaytarır:
        raw_df, clean_df, anomaly_df,
        X_norm, y_norm, normaliser,
        aro_pipeline, regression_metrics,
        regression_coeffs (feature coefficients DataFrame),
        feature_importances (coefficients kimi — uyğunluq üçün)
    """
    clean_df   = filter_data(raw_df)
    anomaly_df = detect_anomalies(clean_df)

    aro = AROPipeline()
    aro.fit(clean_df)

    X_norm, y_norm = aro.normaliser.fit_transform(clean_df)

    coeffs_df = aro.coefficients
    # intercept sütununu çıxar, yalnız feature əmsalları
    feature_cols = [c for c in coeffs_df.columns if c != "intercept"]
    regression_coeffs_simple = coeffs_df[feature_cols].copy()
    regression_coeffs_simple["intercept"] = coeffs_df["intercept"]

    # feature_importances — əmsalların mütləq dəyərlərindən — GB_ prefiksi ilə
    # app.py-da köhnə GB_ prefiksi gözlənilir, uyğunluq üçün saxlayırıq
    fi_rows = []
    for target in TARGET_COLS:
        row = {"target": target}
        for feat in INPUT_FEATURES:
            val = abs(float(coeffs_df.loc[target, feat]))
            row[f"GB_{feat}"] = round(val, 6)
            row[f"RF_{feat}"] = round(val, 6)  # LR üçün eynidir
        fi_rows.append(row)
    feature_importances_df = pd.DataFrame(fi_rows).set_index("target")

    return {
        "raw_df":              raw_df,
        "clean_df":            clean_df,
        "anomaly_df":          anomaly_df,
        "X_norm":              X_norm,
        "y_norm":              y_norm,
        "normaliser":          aro.normaliser,
        "aro_pipeline":        aro,
        "regression_metrics":  aro.metrics,
        "regression_coeffs":   regression_coeffs_simple,
        "feature_importances": feature_importances_df,
        # Linear regression əmsalları tam DataFrame
        "lr_coefficients":     coeffs_df,
    }


if __name__ == "__main__":
    from data_generator import generate_sensor_data

    raw = generate_sensor_data(n_samples=1440)
    out = run_preprocessing_pipeline(raw)

    print("=== Linear Regression Metrikaları ===")
    print(out["regression_metrics"].to_string())

    print(f"\nAnomaly sayı: {out['anomaly_df']['anomaly_final'].sum()}")

    print("\n=== Feature Əmsalları (normallaşdırılmış) ===")
    print(out["regression_coeffs"].to_string())
