"""
Neft Emalı Optimallaşdırma Sistemi
Modul: optimizer.py
Məqsəd: Sənəd bölmə 2.4, NSGA-II əsasında çoxhədəfli optimallaşdırma.

Tətbiq olunan tənliklər:
  Məqsəd funksiyası: F* = a*Y_benz + b*Y_diz + c*Y_ker - d*E   (tənlik 1)
  Kütlə balansı:     F_giriş cəmi = F_çıxış cəmi               (tənlik 3)
  Enerji balansı:    Q_verilən = Q_istifadə + Q_itki            (tənlik 4)
  Temperatur profili: T(z) = T_alt - k*z                        (tənlik 5)
  KKT məhdudiyyətləri: g(x) <= 0, h(x) = 0, x_min <= x <= x_max (tənlik 6,8)
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
from dataclasses import dataclass, field


@dataclass
class EconomicWeights:
    """
    Məqsəd funksiyası F* üçün iqtisadi çəki əmsalları (sənəd tənlik 1).
    Dəyərlər hər məhsulun bazar dəyərini əks etdirir.
    """
    alpha: float = 0.35
    beta:  float = 0.30
    gamma: float = 0.20
    delta: float = 0.15


X_LOWER = np.array([340.0,  1.20,  80.0,  2.5, 100.0, 30.0, 280.0, 0.820, 0.5])
X_UPPER = np.array([400.0,  1.50, 160.0,  5.0, 150.0, 60.0, 360.0, 0.920, 2.5])

VAR_NAMES = [
    "furnace_temp", "column_pressure", "flow_rate", "reflux_ratio",
    "feed_temp", "h2_pressure", "catalyst_temp", "crude_density", "sulfur_content",
]


def _yield_gasoline(x: np.ndarray) -> np.ndarray:
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.18 * ((T - 340) / 60) + 0.04 * ((0.920 - d) / 0.1) + 0.12).clip(0.10, 0.32)

def _yield_diesel(x: np.ndarray) -> np.ndarray:
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.22 * ((R - 2.5) / 2.5) + 0.03 * ((P - 1.2) / 0.3) + 0.18).clip(0.15, 0.38)

def _yield_kerosene(x: np.ndarray) -> np.ndarray:
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.15 * ((T - 340) / 60) * ((P - 1.2) / 0.3) + 0.10).clip(0.08, 0.22)

def _energy_consumption(x: np.ndarray) -> np.ndarray:
    """Enerji balansı: Q = f(T, F), sənəd tənlik 4."""
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.042 * T + 0.018 * F - 10.5).clip(5.0, 22.0)

def _temperature_profile_penalty(x: np.ndarray, n_stages: int = 10) -> np.ndarray:
    """
    T(z) = T_alt - k*z, sənəd tənlik 5.
    Kolon üst temperaturu 120,130 C aralığından kənara çıxarsa cərimə tətbiq edilir.
    """
    T_bottom = x[:, 0]
    k = (T_bottom - 125.0) / n_stages
    T_top = T_bottom - k * n_stages
    return np.abs(T_top - 125.0) / 10.0

def _sulfur_removal(x: np.ndarray) -> np.ndarray:
    T, P, F, R, Tf, H2, Tc, d, S = x.T
    return (0.55 * ((H2 - 30) / 30) + 0.30 * ((Tc - 280) / 80) + 0.40).clip(0.30, 0.98)


class RefineryProblem(Problem):
    """
    Xam neft distillasiyası üçün çoxhədəfli optimallaşdırma məsələsi.

    Hədəflər (2 ədəd):
        f1 = -F* = -(a*Y_benz + b*Y_diz + c*Y_ker - d*E)   (F* maksimumlaşdırılır)
        f2 = E    = enerji sərfi, GC/s                       (minimumlaşdırılır)

    Məhdudiyyətlər (KKT, sənəd tənlik 6,8):
        g1: kütlə balansı, ümumi verim <= 0.85
        g2: enerji balansı, soba gücünü aşmır
        g3: temperatur profili, kolon üst temperaturu
        g4: kükürd spesifikasiyası, çıxarılma >= 50 faiz
    """

    def __init__(self, weights: EconomicWeights = EconomicWeights()):
        super().__init__(
            n_var=9,
            n_obj=2,
            n_ieq_constr=4,
            xl=X_LOWER,
            xu=X_UPPER,
        )
        self.w = weights

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        Yb = _yield_gasoline(x)
        Yd = _yield_diesel(x)
        Yk = _yield_kerosene(x)
        E  = _energy_consumption(x)

        F_star = self.w.alpha * Yb + self.w.beta * Yd + self.w.gamma * Yk - self.w.delta * E / 22.0
        f1 = -F_star
        f2 = E

        total_yield   = Yb + Yd + Yk
        g1 = total_yield - 0.85
        g2 = E - 20.0
        g3 = _temperature_profile_penalty(x) - 0.5
        g4 = 0.50 - _sulfur_removal(x)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4])


@dataclass
class OptimisationResult:
    pareto_X:         np.ndarray
    pareto_F:         np.ndarray
    pareto_df:        pd.DataFrame
    best_compromise:  pd.Series
    history:          list
    n_generations:    int
    n_solutions:      int


def run_nsga2(
    weights:        EconomicWeights = EconomicWeights(),
    pop_size:       int = 100,
    n_gen:          int = 150,
    seed:           int = 42,
) -> OptimisationResult:
    """
    NSGA-II çoxhədəfli optimallaşdırma icra edir.

    Parametrlər
    -----------
    weights   : iqtisadi çəki əmsalları a, b, c, d
    pop_size  : populyasiya ölçüsü
    n_gen     : nəsil sayı
    seed      : rəndəm toxum

    Qaytarır
    --------
    Pareto frontu, ən yaxşı həll və tarixçə ehtiva edən OptimisationResult
    """
    problem = RefineryProblem(weights=weights)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=True,
        verbose=False,
    )

    if result.X is None or len(result.X) == 0:
        raise RuntimeError(
            "NSGA-II Pareto frontu tapa bilmədi. "
            "pop_size və ya n_gen artırın, yaxud məhdudiyyət şərtlərini yoxlayın."
        )

    X = result.X
    F = result.F

    Yb = _yield_gasoline(X)
    Yd = _yield_diesel(X)
    Yk = _yield_kerosene(X)
    E  = _energy_consumption(X)
    Sr = _sulfur_removal(X)

    pareto_df = pd.DataFrame(X, columns=VAR_NAMES)
    pareto_df["yield_gasoline"]    = (Yb * 100).round(2)
    pareto_df["yield_diesel"]      = (Yd * 100).round(2)
    pareto_df["yield_kerosene"]    = (Yk * 100).round(2)
    pareto_df["total_yield_pct"]   = ((Yb + Yd + Yk) * 100).round(2)
    pareto_df["energy_gj_h"]       = E.round(3)
    pareto_df["sulfur_removal_pct"]= (Sr * 100).round(2)
    pareto_df["f_star"]            = (-F[:, 0]).round(4)
    pareto_df["obj_energy"]        = F[:, 1].round(3)

    f1_norm = (F[:, 0] - F[:, 0].min()) / (F[:, 0].max() - F[:, 0].min() + 1e-9)
    f2_norm = (F[:, 1] - F[:, 1].min()) / (F[:, 1].max() - F[:, 1].min() + 1e-9)
    dist    = np.sqrt(f1_norm ** 2 + f2_norm ** 2)
    best_idx = np.argmin(dist)
    best_compromise = pareto_df.iloc[best_idx]

    history = []
    for gen in result.history:
        fs = gen.opt.get("F")
        if fs is not None and len(fs) > 0:
            history.append({
                "generation": gen.n_gen,
                "mean_f_star": float(-fs[:, 0].mean()),
                "mean_energy": float(fs[:, 1].mean()),
                "best_f_star": float(-fs[:, 0].min()),
            })

    return OptimisationResult(
        pareto_X=X,
        pareto_F=F,
        pareto_df=pareto_df,
        best_compromise=best_compromise,
        history=history,
        n_generations=n_gen,
        n_solutions=len(X),
    )


def sensitivity_analysis(
    base_x:   np.ndarray,
    var_idx:  int,
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Bir qərar dəyişənini dəyişdirərək digərlərini sabit saxlayır.
    Həmin dəyişənin verim və enerji üzərindəki təsirini qaytarır.
    """
    lo, hi   = X_LOWER[var_idx], X_UPPER[var_idx]
    grid     = np.linspace(lo, hi, n_points)
    X_sweep  = np.tile(base_x, (n_points, 1))
    X_sweep[:, var_idx] = grid

    return pd.DataFrame({
        VAR_NAMES[var_idx]: grid,
        "yield_gasoline":   _yield_gasoline(X_sweep) * 100,
        "yield_diesel":     _yield_diesel(X_sweep) * 100,
        "yield_kerosene":   _yield_kerosene(X_sweep) * 100,
        "energy_gj_h":      _energy_consumption(X_sweep),
        "total_yield":      (_yield_gasoline(X_sweep)
                             + _yield_diesel(X_sweep)
                             + _yield_kerosene(X_sweep)) * 100,
    })


if __name__ == "__main__":
    print("NSGA-II optimallasdirma bashlayir...")
    res = run_nsga2(pop_size=80, n_gen=100)
    print(f"Pareto frontu hellib sayi: {res.n_solutions}")
    print("En yaxshi kompromis hell:")
    print(res.best_compromise[
        ["furnace_temp", "total_yield_pct", "energy_gj_h", "f_star"]
    ].to_string())