"""
Neft Emalı Optimallaşdırma Sistemi
Modul: data_generator.py
Məqsəd: Xam neft distillasiyasının fiziki parametrlərinə uyğun
sintetik sensor time-series məlumatı yaratmaq.

Fiziki həddlər (sənəd əsasında):
  Borulu soba çıxış temperaturu: 350,400 C
  Rektifikasiya kolonu təzyiqi:  1.2,1.5 atm
  Desalter giriş temperaturu:    100,150 C
  Kolon üst temperaturu:         120,130 C
  Xam neft sıxlığı:              0.730,1.040 g/cm3
  Kükürd miqdarı sinifləri:      0.5 faiz altı, 0.5,2.0 faiz, 2.0 faiz yuxarı
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


BOUNDS = {
    "furnace_temp":    (340.0, 400.0),
    "column_pressure": (1.20,  1.50),
    "flow_rate":       (80.0,  160.0),
    "reflux_ratio":    (2.5,   5.0),
    "feed_temp":       (100.0, 150.0),
    "h2_pressure":     (30.0,  60.0),
    "catalyst_temp":   (280.0, 360.0),
    "crude_density":   (0.820, 0.920),
    "sulfur_content":  (0.5,   2.5),
}

NOISE_SIGMA = {
    "furnace_temp":    0.8,
    "column_pressure": 0.008,
    "flow_rate":       0.6,
    "reflux_ratio":    0.04,
    "feed_temp":       0.5,
    "h2_pressure":     0.3,
    "catalyst_temp":   0.7,
    "crude_density":   0.001,
    "sulfur_content":  0.02,
}


def _compute_yields(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Fraksiya verimləri empirik modellərlə hesablanır.

    Benzin verimi: soba temperaturu artdıqca, sıxlıq azaldıqca artır.
    Dizel verimi: refluks nisbəti və kolon təzyiqindən asılıdır.
    Kerosin verimi: orta temperatur fraksiyasıdır.
    Enerji sərfi: temperatur və axın sürətinin funksiyasıdır (tənlik 4).
    Kükürd çıxarılması: H2 təzyiqi və katalizator temperaturundan asılıdır.

    Parametrlər
    -----------
    rng : np.random.Generator — bütün stokastik əməliyyatlar üçün vahid RNG
    """
    T  = df["furnace_temp"]
    P  = df["column_pressure"]
    F  = df["flow_rate"]
    R  = df["reflux_ratio"]
    Tf = df["feed_temp"]
    H2 = df["h2_pressure"]
    Tc = df["catalyst_temp"]
    Tc = df["catalyst_temp"]
    H2 = df["h2_pressure"]
    d  = df["crude_density"]
    S  = df["sulfur_content"]

    Y_benz = (
        0.11 * ((T - 340) / 60)
        + 0.06 * ((F - 80) / 80)
        + 0.02 * ((0.920 - d) / 0.1)
        + 0.12
        + rng.normal(0, 0.005, len(df))
    ).clip(0.10, 0.32)

    Y_dizel = (
        0.155 * ((R - 2.5) / 2.5)
        + 0.02 * ((P - 1.2) / 0.3)
        + 0.20
        + rng.normal(0, 0.005, len(df))
    ).clip(0.15, 0.38)

    Y_ker = (
        0.12 * ((T - 340) / 60) * ((P - 1.2) / 0.3)
        + 0.04 * ((Tf - 100) / 50)
        + 0.10
        + rng.normal(0, 0.004, len(df))
    ).clip(0.08, 0.26)

    E = (
        0.042 * T
        + 0.018 * F
        + 0.008 * Tf
        + 0.800 * R
        + 0.060 * H2
        + 0.020 * Tc
        + 0.500 * (d - 0.820)
        - 20.92
        + rng.normal(0, 0.3, len(df))
    ).clip(5.0, 22.0)

    S_removal = (
        0.35 * ((H2 - 30) / 30)
        + 0.20 * ((Tc - 280) / 80)
        + 0.40
        + rng.normal(0, 0.02, len(df))
    ).clip(0.30, 0.98)

    df = df.copy()
    df["yield_gasoline"]  = Y_benz.round(4)
    df["yield_diesel"]    = Y_dizel.round(4)
    df["yield_kerosene"]  = Y_ker.round(4)
    df["energy_gj_h"]     = E.round(3)
    df["sulfur_removal"]  = S_removal.round(4)
    df["total_yield"] = (Y_benz + Y_dizel + Y_ker).clip(0, 0.85).round(4)

    return df


def _inject_anomalies(
    df: pd.DataFrame,
    rng: np.random.Generator,
    anomaly_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Sənəd bölmə 2.3: sürşmə, ani dəyişim və çatışmayan dəyər
    növlərindən ibarət sensor anomaliyaları əlavə edilir.

    Parametrlər
    -----------
    rng : np.random.Generator — generate_sensor_data-dan ötürülür,
          eyni seed ilə tam təkrarlana bilən nəticə təmin edir.
    """
    df = df.copy()
    n  = len(df)

    cols = ["furnace_temp", "column_pressure", "flow_rate"]
    for col in cols:
        n_spikes = max(1, int(n * anomaly_rate * 0.5))
        idx = rng.choice(n, n_spikes, replace=False)
        df.loc[df.index[idx], col] *= rng.uniform(1.08, 1.15, n_spikes)

        # Spike dəyərlərini fiziki həddlər daxilinə qaytarır
        lo, hi = BOUNDS[col]
        df[col] = df[col].clip(lo, hi)

        n_missing = max(1, int(n * anomaly_rate * 0.3))
        idx2 = rng.choice(n, n_missing, replace=False)
        df.loc[df.index[idx2], col] = np.nan

    df["is_anomaly"] = False
    for col in cols:
        q1 = df[col].quantile(0.01)
        q3 = df[col].quantile(0.99)
        mask = (df[col] < q1) | (df[col] > q3) | df[col].isna()
        df.loc[mask, "is_anomaly"] = True

    return df


def generate_sensor_data(
    n_samples: int = 1440,
    start: str = "2024-01-01 00:00",
    freq: str = "1min",
    seed: int = 42,
    add_anomalies: bool = True,
) -> pd.DataFrame:
    """
    Sintetik sensor time-series məlumatı yaradır.

    Parametrlər
    -----------
    n_samples    : nümunə sayı (standart 1440, yəni 24 saat, dəqiqəlik interval)
    start        : başlanğıc tarix-vaxt
    freq         : pandas tezlik sətri
    seed         : təkrarlanabilirlik üçün rəndəm toxum
    add_anomalies: sensor anomaliyaları əlavə et

    Qaytarır
    --------
    pd.DataFrame, sütunlar:
        timestamp, furnace_temp, column_pressure, flow_rate, reflux_ratio,
        feed_temp, h2_pressure, catalyst_temp, crude_density, sulfur_content,
        yield_gasoline, yield_diesel, yield_kerosene, energy_gj_h,
        sulfur_removal, total_yield, is_anomaly
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=start, periods=n_samples, freq=freq)

    t = np.linspace(0, 4 * np.pi, n_samples)

    def _signal(lo, hi, phase=0.0, noise_key=None):
        mid   = (lo + hi) / 2
        amp   = (hi - lo) * 0.25
        base  = mid + amp * np.sin(t + phase)
        noise = NOISE_SIGMA.get(noise_key, 0.01) if noise_key else 0
        return base + rng.normal(0, noise, n_samples)

    data = {
        "timestamp":       timestamps,
        "furnace_temp":    _signal(*BOUNDS["furnace_temp"],    0.0, "furnace_temp"),
        "column_pressure": _signal(*BOUNDS["column_pressure"], 1.0, "column_pressure"),
        "flow_rate":       _signal(*BOUNDS["flow_rate"],       2.0, "flow_rate"),
        "reflux_ratio":    _signal(*BOUNDS["reflux_ratio"],    0.5, "reflux_ratio"),
        "feed_temp":       _signal(*BOUNDS["feed_temp"],       1.5, "feed_temp"),
        "h2_pressure":     _signal(*BOUNDS["h2_pressure"],     3.0, "h2_pressure"),
        "catalyst_temp":   _signal(*BOUNDS["catalyst_temp"],   2.5, "catalyst_temp"),
        "crude_density":   _signal(*BOUNDS["crude_density"],   0.3, "crude_density"),
        "sulfur_content":  _signal(*BOUNDS["sulfur_content"],  1.2, "sulfur_content"),
    }

    for key, (lo, hi) in BOUNDS.items():
        data[key] = np.clip(data[key], lo, hi)

    df = pd.DataFrame(data)
    df = _compute_yields(df, rng=rng)

    if add_anomalies:
        df = _inject_anomalies(df, rng=rng)

    return df


def get_operating_summary(df: pd.DataFrame) -> dict:
    """Dashboard KPI kartları üçün əsas əməliyyat statistikası qaytarır."""
    clean = df.dropna()
    return {
        "avg_furnace_temp":    clean["furnace_temp"].mean(),
        "avg_yield_gasoline":  clean["yield_gasoline"].mean() * 100,
        "avg_yield_diesel":    clean["yield_diesel"].mean() * 100,
        "avg_yield_kerosene":  clean["yield_kerosene"].mean() * 100,
        "avg_total_yield":     clean["total_yield"].mean() * 100,
        "avg_energy":          clean["energy_gj_h"].mean(),
        "avg_sulfur_removal":  clean["sulfur_removal"].mean() * 100,
        "n_anomalies":         int(df["is_anomaly"].sum()),
        "anomaly_rate_pct":    df["is_anomaly"].mean() * 100,
        "n_samples":           len(df),
    }


if __name__ == "__main__":
    df = generate_sensor_data(n_samples=1440)
    print(f"Yaradildi: {len(df)} numune")
    print(df.describe().round(3))
    summary = get_operating_summary(df)
    print("\nEmeliyyat xulasesi:")
    for k, v in summary.items():
        print(f"  {k:30s}: {v:.3f}")
