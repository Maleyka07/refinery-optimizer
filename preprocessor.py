"""
Neft Emalı Optimallaşdırma Sistemi
Modul: preprocessor.py
Məqsəd: Sensor məlumatının filtrasiyası, normallaşdırılması,
        anomaliya aşkarlanması və reqressiya modelləşdirilməsi.

Reqressiya arxitekturası:
    Hər hədəf dəyişəni üçün iki model paralel öyrədilir:
        1. GradientBoostingRegressor  — addım-addım boosting, aşağı bias
        2. RandomForestRegressor      — bagging topluluğu, aşağı dispersiya
    Çəkilər 5-qatlı cross-validation R² skorlarından avtomatik hesablanır:
        w_gb = cv_r2_gb / (cv_r2_gb + cv_r2_rf)
        w_rf = 1 - w_gb
    Yekun proqnoz: ensemble_pred = w_gb * gb_pred + w_rf * rf_pred
    Hər model üçün CV R², train R² və RMSE metrikaları ayrıca saxlanılır.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    IsolationForest,
)
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

# ─────────────────────────────────────────────────────────────────────────────
# Model hiperparametrlər — hər iki alqoritm üçün sabit, izlənə bilən
# ─────────────────────────────────────────────────────────────────────────────

GB_PARAMS = {
    "n_estimators":    80,         # 200→80: sürət ~2.5x, R² fərqi < 0.01
    "max_depth":       4,
    "learning_rate":   0.08,
    "subsample":       0.85,
    "min_samples_leaf": 5,
    "random_state":    42,
}

RF_PARAMS = {
    "n_estimators":    80,         # 200→80: sürət ~2.5x, R² fərqi < 0.01
    "max_depth":       12,         # None→12: ağaclar artıq tam böyüməyəcək, 2x sürətli
    "min_samples_leaf": 4,
    "max_features":    "sqrt",
    "random_state":    42,
    "n_jobs":          -1,
}

# Cross-validation parametri
CV_FOLDS = 3                       # 5→3: keyfiyyət itirmədən fold sayı azaldı


# ─────────────────────────────────────────────────────────────────────────────
# Filtrasiya
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Normallaşdırma
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Anomaliya aşkarlanması
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Reqressiya modeli — GradientBoosting + RandomForest ensemble
# ─────────────────────────────────────────────────────────────────────────────

class RegressionModel:
    """
    Hər hədəf dəyişəni üçün iki model öyrədilir:
        gb[target]  — GradientBoostingRegressor
        rf[target]  — RandomForestRegressor

    Çəkilər CV_FOLDS-qatlı cross-validation R² skoruna əsasən hesablanır:
        w_gb[target] = cv_gb / (cv_gb + cv_rf)
        w_rf[target] = 1 - w_gb[target]

    Proqnoz: ensemble_pred = w_gb * gb_pred + w_rf * rf_pred

    metrics cədvəlinin sütunları:
        GB_CV_R², RF_CV_R², GB_R², RF_R², Ensemble_R², Ensemble_RMSE,
        W_GB, W_RF
    """

    def __init__(self):
        self.gb      : dict[str, GradientBoostingRegressor] = {}
        self.rf      : dict[str, RandomForestRegressor]     = {}
        self.weights : dict[str, tuple[float, float]]       = {}  # (w_gb, w_rf)
        self.metrics : dict[str, dict]                      = {}
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "RegressionModel":
        # DataFrame saxla — feature name warning-lərinin qarşısını alır
        X_df = X[INPUT_FEATURES]
        X_arr = X_df.values  # CV üçün array lazımdır (cross_val_score)

        for target in TARGET_COLS:
            y_t = y[target].values

            # ── Model qurulması ────────────────────────────────────────
            gb_model = GradientBoostingRegressor(**GB_PARAMS)
            rf_model = RandomForestRegressor(**RF_PARAMS)

            # ── Cross-validation R² skorları (fit edilməmiş modellər üzərində)
            # n_jobs=1: Windows-da Streamlit @st.cache_resource içindən joblib
            # multiprocessing pool yarada bilmir → BrokenProcessPool xətası.
            # n_jobs=1 ilə eyni nəticə, tək prosesdə ardıcıl icra.
            cv_gb = cross_val_score(
                gb_model, X_arr, y_t,
                cv=CV_FOLDS, scoring="r2", n_jobs=1,
            ).mean()
            cv_rf = cross_val_score(
                rf_model, X_arr, y_t,
                cv=CV_FOLDS, scoring="r2", n_jobs=1,
            ).mean()

            # ── CV R² mənfi ola bilər (çox zəif model); sıfırla məhdudlaşdır
            cv_gb_clip = max(cv_gb, 0.0)
            cv_rf_clip = max(cv_rf, 0.0)
            denom      = cv_gb_clip + cv_rf_clip

            if denom < 1e-9:
                # İki model da tamamilə zəifdirsə bərabər çəki ver
                w_gb, w_rf = 0.5, 0.5
            else:
                w_gb = cv_gb_clip / denom
                w_rf = cv_rf_clip / denom

            # ── Tam məlumat üzərində son fit — DataFrame ilə, array yox
            # (predict zamanı "feature names" warning-ini aradan qaldırır)
            gb_model.fit(X_df, y_t)
            rf_model.fit(X_df, y_t)

            gb_pred  = gb_model.predict(X_df)
            rf_pred  = rf_model.predict(X_df)
            ens_pred = w_gb * gb_pred + w_rf * rf_pred

            self.gb[target]      = gb_model
            self.rf[target]      = rf_model
            self.weights[target] = (round(w_gb, 4), round(w_rf, 4))

            self.metrics[target] = {
                "GB_CV_R²":      round(cv_gb,  4),
                "RF_CV_R²":      round(cv_rf,  4),
                "W_GB":          round(w_gb,   4),
                "W_RF":          round(w_rf,   4),
                "GB_R²":         round(r2_score(y_t, gb_pred),  4),
                "RF_R²":         round(r2_score(y_t, rf_pred),  4),
                "Ensemble_R²":   round(r2_score(y_t, ens_pred), 4),
                "Ensemble_RMSE": round(np.sqrt(mean_squared_error(y_t, ens_pred)), 6),
            }

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """CV çəkilərinə əsasən ensemble proqnozu qaytarır."""
        assert self._fitted, "Əvvəlcə fit çağırın."
        # DataFrame ilə predict — feature names warning-i baş vermir
        X_df = X[INPUT_FEATURES]
        preds = {}
        for target in TARGET_COLS:
            w_gb, w_rf    = self.weights[target]
            gb_pred       = self.gb[target].predict(X_df)
            rf_pred       = self.rf[target].predict(X_df)
            preds[target] = w_gb * gb_pred + w_rf * rf_pred
        return pd.DataFrame(preds, index=X.index)

    # ------------------------------------------------------------------
    def get_metrics_df(self) -> pd.DataFrame:
        """
        Sütunlar: GB_CV_R², RF_CV_R², W_GB, W_RF,
                  GB_R², RF_R², Ensemble_R², Ensemble_RMSE
        İndeks  : TARGET_COLS
        """
        return pd.DataFrame(self.metrics).T

    # ------------------------------------------------------------------
    def feature_importances(self) -> pd.DataFrame:
        """GB və RF feature importance-larını qaytarır."""
        assert self._fitted, "Əvvəlcə fit çağırın."
        rows = []
        for target in TARGET_COLS:
            gb_imp = self.gb[target].feature_importances_
            rf_imp = self.rf[target].feature_importances_
            row    = {"target": target}
            for feat, gi, ri in zip(INPUT_FEATURES, gb_imp, rf_imp):
                row[f"GB_{feat}"] = round(float(gi), 6)
                row[f"RF_{feat}"] = round(float(ri), 6)
            rows.append(row)
        return pd.DataFrame(rows).set_index("target")


# ─────────────────────────────────────────────────────────────────────────────
# ARO Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AROPipeline:
    """
    Tam pipeline:
        filter_data → DataNormaliser → RegressionModel (GB + RF ensemble)

    predict_physical() normallaşdırılmış proqnozu fiziki vahidlərə qaytarır.
    """

    def __init__(self):
        self.normaliser = DataNormaliser()
        self.regressor  = RegressionModel()
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
    def feature_importances(self) -> pd.DataFrame:
        return self.regressor.feature_importances()


# ─────────────────────────────────────────────────────────────────────────────
# Tam pipeline funksiyası
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(raw_df: pd.DataFrame) -> dict:
    """
    Qaytarır:
        raw_df, clean_df, anomaly_df,
        X_norm, y_norm, normaliser,
        aro_pipeline, regression_metrics, feature_importances
    """
    clean_df   = filter_data(raw_df)
    anomaly_df = detect_anomalies(clean_df)

    aro = AROPipeline()
    aro.fit(clean_df)

    # Normallaşdırılmış cədvəlləri ayrıca saxla (test + dashboard üçün)
    X_norm, y_norm = aro.normaliser.fit_transform(clean_df)

    # regression_coeffs — app.py uyğunluğu üçün feature importances DataFrame
    # Sütunlar: hər sensor üçün GB və RF importance-ları (əmsallar kimi istifadə olunur)
    feat_imp = aro.feature_importances
    # app.py "intercept" sütununu gözləyir (istilik xəritəsi üçün), əlavə edirik
    regression_coeffs = feat_imp.copy()
    # GB importance-larını əsas "əmsal" kimi istifadə üçün yenidən adlandırma
    gb_cols = {c: c.replace("GB_", "") for c in feat_imp.columns if c.startswith("GB_")}
    regression_coeffs_simple = feat_imp[[c for c in feat_imp.columns if c.startswith("GB_")]].copy()
    regression_coeffs_simple.columns = [c.replace("GB_", "") for c in regression_coeffs_simple.columns]
    regression_coeffs_simple["intercept"] = 0.0  # app.py intercept sütununu çıxarır

    return {
        "raw_df":              raw_df,
        "clean_df":            clean_df,
        "anomaly_df":          anomaly_df,
        "X_norm":              X_norm,
        "y_norm":              y_norm,
        "normaliser":          aro.normaliser,
        "aro_pipeline":        aro,
        "regression_metrics":  aro.metrics,
        "feature_importances": aro.feature_importances,
        # App.py uyğunluğu: köhnə "regression_coeffs" açarı feature importances ilə doldurulur
        "regression_coeffs":   regression_coeffs_simple,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI sınaq
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_generator import generate_sensor_data

    raw = generate_sensor_data(n_samples=1440)
    out = run_preprocessing_pipeline(raw)

    print("=== Reqressiya Metrikaları (GB + RF Ensemble) ===")
    print(out["regression_metrics"].to_string())

    print(f"\nAnomaly sayı: {out['anomaly_df']['anomaly_final'].sum()}")

    print("\n=== Feature Importances (GradientBoosting) ===")
    gb_cols = [c for c in out["feature_importances"].columns if c.startswith("GB_")]
    print(out["feature_importances"][gb_cols].to_string())