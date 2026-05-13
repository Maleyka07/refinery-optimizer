"""
Neft Emalı Optimallaşdırma Sistemi
Modul: tests/test_pipeline.py
Məqsəd: data_generator, preprocessor və optimizer modullarının
        əsas davranışlarını avtomatik yoxlamaq.

İcra:
    pip install pytest
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generator import (
    generate_sensor_data,
    get_operating_summary,
    BOUNDS,
)
from preprocessor import (
    filter_data,
    detect_anomalies,
    DataNormaliser,
    RegressionModel,
    AROPipeline,
    INPUT_FEATURES,
    TARGET_COLS,
)


# ─────────────────────────────────────────────────────────────
# Köməkçi fixture-lər
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """225 nümunəlik kiçik dataset — testlər üçün sürətlidir."""
    return generate_sensor_data(n_samples=225, seed=42)


@pytest.fixture(scope="module")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return filter_data(raw_df)


@pytest.fixture(scope="module")
def fitted_aro(clean_df: pd.DataFrame) -> AROPipeline:
    aro = AROPipeline()
    aro.fit(clean_df)
    return aro


# ─────────────────────────────────────────────────────────────
# 1. data_generator testləri
# ─────────────────────────────────────────────────────────────

class TestDataGenerator:

    def test_shape(self, raw_df):
        """Cədvəl ölçüsü gözlənilənə uyğun olmalıdır."""
        assert len(raw_df) == 225
        assert "timestamp" in raw_df.columns
        assert "is_anomaly" in raw_df.columns

    def test_expected_columns(self, raw_df):
        """Bütün sensor sütunları mövcud olmalıdır."""
        for col in INPUT_FEATURES:
            assert col in raw_df.columns, f"Sütun yoxdur: {col}"

    def test_physical_bounds(self, raw_df):
        """NaN olmayan dəyərlər fiziki həddlər daxilində qalmalıdır."""
        clean = raw_df.dropna()
        for col, (lo, hi) in BOUNDS.items():
            assert (clean[col] >= lo).all(), f"{col} alt həddin altına düşüb"
            assert (clean[col] <= hi).all(), f"{col} üst həddi aşıb"

    def test_reproducibility(self):
        """Eyni seed ilə iki çağırış eyni nəticə verməlidir."""
        df1 = generate_sensor_data(n_samples=100, seed=7)
        df2 = generate_sensor_data(n_samples=100, seed=7)
        pd.testing.assert_frame_equal(
            df1.drop(columns="timestamp"),
            df2.drop(columns="timestamp"),
        )

    def test_different_seeds_differ(self):
        """Fərqli seed-lər fərqli data yaratmalıdır."""
        df1 = generate_sensor_data(n_samples=100, seed=1)
        df2 = generate_sensor_data(n_samples=100, seed=99)
        assert not df1["furnace_temp"].equals(df2["furnace_temp"])

    def test_anomaly_flag_exists(self, raw_df):
        """Anomaliya bayrağı bool tipli olmalıdır."""
        assert raw_df["is_anomaly"].dtype == bool

    def test_anomaly_rate_reasonable(self, raw_df):
        """Anomaliya faizi sənaye norması olan 1.5–3.0 aralığında olmalıdır."""
        rate = raw_df["is_anomaly"].mean() * 100
        assert 0.5 <= rate <= 10.0, f"Anomaliya faizi gözlənilməz: {rate:.2f}%"

    def test_yield_columns_in_range(self, raw_df):
        """Verim dəyərləri 0–1 aralığında olmalıdır."""
        for col in ["yield_gasoline", "yield_diesel", "yield_kerosene", "total_yield"]:
            assert (raw_df[col].dropna() >= 0).all()
            assert (raw_df[col].dropna() <= 1).all()

    def test_operating_summary_keys(self, raw_df):
        """Əməliyyat xülasəsi bütün açarları ehtiva etməlidir."""
        summary = get_operating_summary(raw_df)
        expected_keys = [
            "avg_furnace_temp", "avg_yield_gasoline", "avg_total_yield",
            "avg_energy", "n_anomalies", "anomaly_rate_pct", "n_samples",
        ]
        for k in expected_keys:
            assert k in summary, f"Açar yoxdur: {k}"


# ─────────────────────────────────────────────────────────────
# 2. preprocessor testləri
# ─────────────────────────────────────────────────────────────

class TestPreprocessor:

    def test_filter_removes_nan(self, raw_df):
        """Filtrasiyadan sonra INPUT_FEATURES-də NaN qalmamalıdır."""
        cleaned = filter_data(raw_df)
        assert cleaned[INPUT_FEATURES].isna().sum().sum() == 0

    def test_filter_preserves_most_rows(self, raw_df, clean_df):
        """Filtrasiya nümunələrin 80%-dən çoxunu saxlamalıdır."""
        assert len(clean_df) >= len(raw_df) * 0.80

    def test_normaliser_range(self, clean_df):
        """Min-Maks normallaşdırma sonrası bütün dəyərlər [0, 1] aralığında olmalıdır."""
        norm = DataNormaliser()
        X_norm, _ = norm.fit_transform(clean_df)
        assert X_norm.min().min() >= -1e-9
        assert X_norm.max().max() <= 1 + 1e-9

    def test_normaliser_inverse_roundtrip(self, clean_df):
        """Normallaşdırma → tərs çevrilmə orijinal dəyərləri qaytarmalıdır."""
        norm = DataNormaliser()
        X_norm, y_norm = norm.fit_transform(clean_df)
        y_recovered = norm.inverse_transform_y(y_norm.values)
        original = clean_df[TARGET_COLS].values
        np.testing.assert_allclose(y_recovered, original, atol=1e-6)

    def test_anomaly_detection_columns(self, clean_df):
        """Anomaliya aşkarlaması tələb olunan sütunları yaratmalıdır."""
        anom_df = detect_anomalies(clean_df)
        for col in ["anomaly_iqr", "anomaly_iforest", "anomaly_final", "anomaly_score"]:
            assert col in anom_df.columns, f"Sütun yoxdur: {col}"

    def test_regression_r2_threshold(self, fitted_aro):
        """ARO ensemble modelinin Ensemble_R² >= 0.85 olmalıdır."""
        metrics = fitted_aro.metrics
        for target in TARGET_COLS:
            r2 = metrics.loc[target, "Ensemble_R²"]
            assert r2 >= 0.85, (
                f"{target} üçün Ensemble R² = {r2:.4f} — 0.85 həddinin altındadır"
            )

    def test_cv_weights_sum_to_one(self, fitted_aro):
        """Hər hədəf üçün W_GB + W_RF = 1.0 olmalıdır."""
        metrics = fitted_aro.metrics
        for target in TARGET_COLS:
            total = metrics.loc[target, "W_GB"] + metrics.loc[target, "W_RF"]
            assert abs(total - 1.0) < 1e-6, (
                f"{target} üçün çəkilər cəmi {total:.6f} — 1.0 deyil"
            )

    def test_aro_predict_shape(self, fitted_aro, clean_df):
        """ARO proqnozu doğru forma qaytarmalıdır."""
        preds = fitted_aro.predict_physical(clean_df.head(10))
        assert preds.shape == (10, len(TARGET_COLS))

    def test_aro_not_fitted_raises(self):
        """Fit edilməmiş pipeline predict çağrısında xəta verməlidir."""
        aro = AROPipeline()
        with pytest.raises(AssertionError):
            dummy = pd.DataFrame(
                np.zeros((5, len(INPUT_FEATURES))), columns=INPUT_FEATURES
            )
            aro.predict_physical(dummy)


# ─────────────────────────────────────────────────────────────
# 3. optimizer testləri  (yüngül parametrlərlə)
# ─────────────────────────────────────────────────────────────

class TestOptimizer:
    """
    NSGA-II testləri kiçik pop_size/n_gen ilə icra olunur
    ki CI ortamında sürətli işləsin (~ 5–10 san).
    """

    @pytest.fixture(scope="class")
    def opt_result(self):
        from optimizer import run_nsga2, EconomicWeights
        return run_nsga2(pop_size=20, n_gen=15, seed=42)

    def test_pareto_not_empty(self, opt_result):
        """Pareto frontu ən azı 1 həll ehtiva etməlidir."""
        assert opt_result.n_solutions >= 1

    def test_pareto_shape(self, opt_result):
        """Pareto X matrisin sütun sayı 9 dəyişənə uyğun olmalıdır."""
        assert opt_result.pareto_X.shape[1] == 9

    def test_pareto_bounds_respected(self, opt_result):
        """Pareto həllərinin hamısı fiziki həddlər daxilindədir."""
        from optimizer import X_LOWER, X_UPPER
        assert (opt_result.pareto_X >= X_LOWER - 1e-6).all()
        assert (opt_result.pareto_X <= X_UPPER + 1e-6).all()

    def test_best_compromise_has_expected_fields(self, opt_result):
        """Ən yaxşı kompromis həll gözlənilən sütunları ehtiva etməlidir."""
        bc = opt_result.best_compromise
        for field in ["furnace_temp", "total_yield_pct", "energy_gj_h", "f_star"]:
            assert field in bc.index, f"Alan yoxdur: {field}"

    def test_f_star_positive(self, opt_result):
        """F* məqsəd funksiyası müsbət olmalıdır (verim > normallaşdırılmış enerji)."""
        assert (opt_result.pareto_df["f_star"] > 0).any()

    def test_energy_within_physical_range(self, opt_result):
        """Enerji sərfi 5–22 GJ/h fiziki aralığında qalmalıdır."""
        e = opt_result.pareto_df["energy_gj_h"]
        assert (e >= 4.9).all() and (e <= 22.1).all()

    def test_history_length(self, opt_result):
        """Tarixçə nəsil sayına bərabər uzunluqda olmalıdır."""
        assert len(opt_result.history) == opt_result.n_generations

    def test_runtime_error_on_empty_result(self, monkeypatch):
        """result.X = None olduqda RuntimeError qaldırılmalıdır."""
        from optimizer import run_nsga2
        import optimizer as opt_module
        from pymoo.optimize import minimize as orig_minimize

        class _FakeResult:
            X = None
            F = None
            history = []

        monkeypatch.setattr(opt_module, "minimize", lambda *a, **kw: _FakeResult())
        with pytest.raises(RuntimeError, match="Pareto frontu tapa bilmədi"):
            run_nsga2(pop_size=5, n_gen=3, seed=0)