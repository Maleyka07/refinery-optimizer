"""
Neft Emalı Optimallaşdırma Sistemi
Modul: optimizer.py
Məqsəd: Sənəd bölmə 2.4, NSGA-II əsasında çoxhədəfli optimallaşdırma.

Məqsəd funksiyası (Variant B):
  F* = α·Yb + β·Yd + γ·Yk − δ·(E/E_max) + ε·S_removal

Tənlik dəyişənlərinin F* və E üzərindəki tradeoff rolu:
  furnace_temp  : Yb↑ Yk↑ F*↑  /  E↑  → tradeoff ✓
  flow_rate     : Yb↑ F*↑       /  E↑  → tradeoff ✓
  reflux_ratio  : Yd↑ F*↑       /  E↑  → tradeoff ✓
  feed_temp     : Yk↑ F*↑       /  E↑  → tradeoff ✓
  h2_pressure   : Sr↑ F*↑       /  E↑  → tradeoff ✓
  catalyst_temp : Sr↑ F*↑       /  E↑  → tradeoff ✓
  crude_density : Yb↓ F*↓       /  E↑  → ikitərəfli cərimə
  column_pressure: Yd↑Yk↑ F*↑  /  E=0  → F*↑ üçün max seçilir (fiziki doğrudur)
  sulfur_content : F*=0, E=0    → nəzarətsiz dəyişən (xammal xassəsi)
"""

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from dataclasses import dataclass


@dataclass
class EconomicWeights:
    """
    F* = α·Yb + β·Yd + γ·Yk − δ·(E/E_max) + ε·S_removal

    alpha   : benzin fraksiyasının iqtisadi çəkisi
    beta    : dizel fraksiyasının iqtisadi çəkisi
    gamma   : kerosin fraksiyasının iqtisadi çəkisi
    delta   : enerji sərfinin cərimə çəkisi
    epsilon : kükürd çıxarılmasının mükafat çəkisi
    """
    alpha:   float = 0.35
    beta:    float = 0.30
    gamma:   float = 0.20
    delta:   float = 0.15
    epsilon: float = 0.10


X_LOWER = np.array([340.0, 1.20,  80.0, 2.5, 100.0, 30.0, 280.0, 0.820, 0.5])
X_UPPER = np.array([400.0, 1.50, 160.0, 5.0, 150.0, 60.0, 360.0, 0.920, 2.5])

VAR_NAMES = [
    "furnace_temp", "column_pressure", "flow_rate", "reflux_ratio",
    "feed_temp", "h2_pressure", "catalyst_temp", "crude_density", "sulfur_content",
]

VAR_LABELS = {
    "furnace_temp":    "Soba temp. (°C)",
    "column_pressure": "Kolon təzyiqi (atm)",
    "flow_rate":       "Axın sürəti (m³/s)",
    "reflux_ratio":    "Refluks nisbəti",
    "feed_temp":       "Qidalanma temp. (°C)",
    "h2_pressure":     "H₂ təzyiqi (bar)",
    "catalyst_temp":   "Katalizator temp. (°C)",
    "crude_density":   "Neft sıxlığı (q/sm³)",
    "sulfur_content":  "Kükürd miqdarı (%)",
}

E_MAX = 22.0


# ── Verim funksiyaları ──────────────────────────────────────────────────────

def _yield_gasoline(x: np.ndarray) -> np.ndarray:
    """
    Yb = 0.18·((T−340)/60) + 0.05·((F−80)/80) + 0.04·((0.920−d)/0.1) + 0.12
    Aralıq: [0.120, 0.310] — clip [0.10, 0.32] aktivləşmir.
    flow_rate artan throughput vasitəsilə Yb-nı artırır.
    """
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.11*((T - 340)/60)
            + 0.06*((F - 80)/80)
            + 0.02*((0.920 - d)/0.1)
            + 0.12).clip(0.10, 0.32)


def _yield_diesel(x: np.ndarray) -> np.ndarray:
    """
    Yd = 0.155·((R−2.5)/2.5) + 0.02·((P−1.2)/0.3) + 0.20
    Aralıq: [0.200, 0.375] — clip [0.15, 0.38] aktivləşmir.
    """
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.155*((R - 2.5)/2.5)
            + 0.02*((P - 1.2)/0.3)
            + 0.20).clip(0.15, 0.38)


def _yield_kerosene(x: np.ndarray) -> np.ndarray:
    """
    Yk = 0.12·((T−340)/60)·((P−1.2)/0.3) + 0.04·((Tf−100)/50) + 0.10
    feed_temp daxildir: yüksək Tf orta fraksiya keçişini artırır.
    Aralıq: [0.100, 0.260] — clip [0.08, 0.22] yuxarıda aktivdir.
    """
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.12*((T - 340)/60)*((P - 1.2)/0.3)
            + 0.04*((Tf - 100)/50)
            + 0.10).clip(0.08, 0.26)


def _energy_consumption(x: np.ndarray) -> np.ndarray:
    """
    E = 0.042·T + 0.018·F + 0.008·Tf + 0.006·R + 0.003·H2 + 0.002·Tc
        + 0.5·(d−0.820) − 11.5   [GJ/s]

    Genişləndirilmiş enerji balansı (tənlik 4):
      T   — sobanın əsas enerji istehlakı
      F   — axın sürəti enerji tələbini artırır
      Tf  — aşağı qidalanma temperaturu sobanın yükünü artırır
      R   — refluks döngəsü böyük istilik tələb edir (əmsal: 0.800)
      H2  — HDS reaktorunda H₂ kompressiya enerjisi (əmsal: 0.060)
      Tc  — katalizator temperaturunun saxlanması (əmsal: 0.030)
      d   — ağır neft daha çox enerji tələb edir (əmsal: 0.500)

    Aralıq: [5.00, 15.61] GJ/s — clip [5.0, 22.0] yalnız minimumda aktivdir.
    """
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.042*T
            + 0.018*F
            + 0.008*Tf
            + 0.800*R
            + 0.060*H2
            + 0.020*Tc
            + 0.500*(d - 0.820)
            - 20.92).clip(5.0, 22.0)


def _temperature_profile_penalty(x: np.ndarray, n_stages: int = 10) -> np.ndarray:
    """T(z) = T_alt − k·z (tənlik 5). Kolon üst temp. 120–130°C şərti."""
    T_bottom = x[:, 0]
    k        = (T_bottom - 125.0) / n_stages
    T_top    = T_bottom - k * n_stages
    return np.abs(T_top - 125.0) / 10.0


def _sulfur_removal(x: np.ndarray) -> np.ndarray:
    """
    S = 0.35·((H2−30)/30) + 0.20·((Tc−280)/80) + 0.40
    Aralıq: [0.400, 0.950] — clip [0.30, 0.98] aktivləşmir.
    """
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.35*((H2 - 30)/30)
            + 0.20*((Tc - 280)/80)
            + 0.40).clip(0.30, 0.98)


# ── Optimallaşdırma məsələsi ────────────────────────────────────────────────

class RefineryProblem(Problem):
    """
    Hədəflər (2):
      f1 = −F* → minimize  (F* maksimumlaşdırılır)
      f2 =  E  → minimize

    Məhdudiyyətlər KKT (tənliklər 6–8):
      g1: Yb+Yd+Yk ≤ 0.85
      g2: E ≤ 20 GJ/s
      g3: temperatur profili cəriməsi ≤ 0.5
      g4: Sr ≥ 0.50
    """

    def __init__(self, weights: EconomicWeights = EconomicWeights()):
        super().__init__(
            n_var=9, n_obj=2, n_ieq_constr=4,
            xl=X_LOWER, xu=X_UPPER,
        )
        self.w = weights

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        Yb = _yield_gasoline(x)
        Yd = _yield_diesel(x)
        Yk = _yield_kerosene(x)
        E  = _energy_consumption(x)
        Sr = _sulfur_removal(x)

        F_star = (self.w.alpha * Yb
                  + self.w.beta  * Yd
                  + self.w.gamma * Yk
                  - self.w.delta * (E / E_MAX)
                  + self.w.epsilon * Sr)

        out["F"] = np.column_stack([-F_star, E])
        out["G"] = np.column_stack([
            (Yb + Yd + Yk) - 0.85,
            E - 20.0,
            _temperature_profile_penalty(x) - 0.5,
            0.50 - Sr,
        ])


# ── Nəticə dataclass ────────────────────────────────────────────────────────

@dataclass
class OptimisationResult:
    pareto_X:        np.ndarray
    pareto_F:        np.ndarray
    pareto_df:       pd.DataFrame
    best_compromise: pd.Series
    history:         list
    n_generations:   int
    n_solutions:     int


# ── NSGA-II icrası ──────────────────────────────────────────────────────────

def run_nsga2(
    weights:  EconomicWeights = EconomicWeights(),
    pop_size: int = 100,
    n_gen:    int = 150,
    seed:     int = 42,
) -> OptimisationResult:
    problem = RefineryProblem(weights=weights)

    result = minimize(
        problem,
        NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        ),
        get_termination("n_gen", n_gen),
        seed=seed,
        save_history=True,
        verbose=False,
    )

    if result.X is None or len(result.X) == 0:
        raise RuntimeError("NSGA-II Pareto frontu tapa bilmədi.")

    X, F = result.X, result.F
    Yb = _yield_gasoline(X)
    Yd = _yield_diesel(X)
    Yk = _yield_kerosene(X)
    E  = _energy_consumption(X)
    Sr = _sulfur_removal(X)

    pareto_df = pd.DataFrame(X, columns=VAR_NAMES)
    pareto_df["yield_gasoline"]     = (Yb * 100).round(2)
    pareto_df["yield_diesel"]       = (Yd * 100).round(2)
    pareto_df["yield_kerosene"]     = (Yk * 100).round(2)
    pareto_df["total_yield_pct"]    = ((Yb + Yd + Yk) * 100).round(2)
    pareto_df["energy_gj_h"]        = E.round(3)
    pareto_df["sulfur_removal_pct"] = (Sr * 100).round(2)
    pareto_df["f_star"]             = (-F[:, 0]).round(4)
    pareto_df["obj_energy"]         = F[:, 1].round(3)

    f1n = (F[:, 0] - F[:, 0].min()) / (F[:, 0].max() - F[:, 0].min() + 1e-9)
    f2n = (F[:, 1] - F[:, 1].min()) / (F[:, 1].max() - F[:, 1].min() + 1e-9)
    best_compromise = pareto_df.iloc[np.argmin(np.sqrt(f1n**2 + f2n**2))]

    history = []
    for gen in result.history:
        fs = gen.opt.get("F")
        if fs is not None and len(fs) > 0:
            history.append({
                "generation":  gen.n_gen,
                "mean_f_star": float(-fs[:, 0].mean()),
                "mean_energy": float(fs[:, 1].mean()),
                "best_f_star": float(-fs[:, 0].min()),
            })

    return OptimisationResult(
        pareto_X=X, pareto_F=F, pareto_df=pareto_df,
        best_compromise=best_compromise,
        history=history, n_generations=n_gen, n_solutions=len(X),
    )


# ── Həssaslıq analizi ───────────────────────────────────────────────────────

def sensitivity_analysis(
    base_x:   np.ndarray,
    var_idx:  int,
    n_points: int = 50,
) -> pd.DataFrame:
    lo, hi  = X_LOWER[var_idx], X_UPPER[var_idx]
    grid    = np.linspace(lo, hi, n_points)
    X_sweep = np.tile(base_x, (n_points, 1))
    X_sweep[:, var_idx] = grid
    return pd.DataFrame({
        VAR_NAMES[var_idx]: grid,
        "yield_gasoline":   _yield_gasoline(X_sweep)  * 100,
        "yield_diesel":     _yield_diesel(X_sweep)    * 100,
        "yield_kerosene":   _yield_kerosene(X_sweep)  * 100,
        "energy_gj_h":      _energy_consumption(X_sweep),
        "sulfur_removal":   _sulfur_removal(X_sweep)  * 100,
        "total_yield":      (_yield_gasoline(X_sweep)
                             + _yield_diesel(X_sweep)
                             + _yield_kerosene(X_sweep)) * 100,
    })


if __name__ == "__main__":
    res = run_nsga2(pop_size=80, n_gen=100)
    print(f"Pareto həll sayı: {res.n_solutions}")
    df = res.pareto_df
    print("\nDəyişkənlik (min → max):")
    for col in VAR_NAMES + ["total_yield_pct","energy_gj_h","sulfur_removal_pct","f_star"]:
        print(f"  {col:22s}: [{df[col].min():.3f}, {df[col].max():.3f}] Δ={df[col].max()-df[col].min():.3f}")
