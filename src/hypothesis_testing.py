"""
Statistical tests used in Phase 2.

* compare_groups(): Welch two-sample t-test + assumption checks + effect size + non-parametric check.
* ols_hac(): OLS with Newey-West (HAC) standard errors, written with NumPy only, used to
  (a) control for confounders (season, weather, temperature, year) and
  (b) allow for day-to-day autocorrelation, which violates the t-test's independence assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compare_groups(a: pd.Series, b: pd.Series, alpha: float = 0.05,
                   label_a: str = "group A", label_b: str = "group B") -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)

    t = stats.ttest_ind(a, b, equal_var=False)
    se = np.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    diff = ma - mb

    pooled_sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = diff / pooled_sd
    hedges_g = d * (1 - 3 / (4 * (na + nb) - 9))

    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    rank_biserial = 1 - 2 * mw.statistic / (na * nb)   # sign: positive => A tends to be larger

    return {
        "label_a": label_a, "label_b": label_b,
        "n_a": na, "n_b": nb, "mean_a": ma, "mean_b": mb,
        "sd_a": np.sqrt(va), "sd_b": np.sqrt(vb),
        "diff": diff, "ci_low": diff - tcrit * se, "ci_high": diff + tcrit * se,
        "welch_t": t.statistic, "welch_df": df, "p_value": t.pvalue,
        "hedges_g": hedges_g,
        "levene_p": stats.levene(a, b).pvalue,
        "shapiro_p_a": stats.shapiro(a).pvalue, "shapiro_p_b": stats.shapiro(b).pvalue,
        "skew_a": stats.skew(a), "skew_b": stats.skew(b),
        "mannwhitney_p": mw.pvalue, "rank_biserial": -rank_biserial,
        "reject_h0": t.pvalue < alpha, "alpha": alpha,
    }


def format_result(r: dict, value_fmt: str = "{:.3f}") -> pd.DataFrame:
    f = lambda v: value_fmt.format(v)
    rows = [
        (f"n ({r['label_a']} / {r['label_b']})", f"{r['n_a']} / {r['n_b']}"),
        (f"Mean {r['label_a']}", f(r["mean_a"])),
        (f"Mean {r['label_b']}", f(r["mean_b"])),
        ("Difference (A - B)", f(r["diff"])),
        ("95% CI of difference", f"[{f(r['ci_low'])}, {f(r['ci_high'])}]"),
        ("Welch t (df)", f"{r['welch_t']:.2f} ({r['welch_df']:.0f})"),
        ("p-value (Welch, two-sided)", f"{r['p_value']:.2e}"),
        ("Effect size Hedges' g", f"{r['hedges_g']:.2f}"),
        ("Levene p (equal variances?)", f"{r['levene_p']:.3g}"),
        ("Shapiro p (A / B normal?)", f"{r['shapiro_p_a']:.3g} / {r['shapiro_p_b']:.3g}"),
        ("Mann-Whitney U p (robustness)", f"{r['mannwhitney_p']:.2e}"),
        ("Rank-biserial correlation", f"{r['rank_biserial']:.2f}"),
        (f"Decision at alpha = {r['alpha']}", "Reject H0" if r["reject_h0"] else "Fail to reject H0"),
    ]
    return pd.DataFrame(rows, columns=["Statistic", "Value"]).set_index("Statistic")


def ols_hac(y: pd.Series, X: pd.DataFrame, maxlags: int = 7) -> pd.DataFrame:
    """OLS with Newey-West standard errors (Bartlett kernel). Rows must be in time order."""
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    names = ["const"] + list(X.columns)
    yv = np.asarray(y, float)
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    beta = XtX_inv @ Xc.T @ yv
    u = yv - Xc @ beta
    Xu = Xc * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, maxlags + 1):
        w = 1 - lag / (maxlags + 1)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    n, k = Xc.shape
    cov = XtX_inv @ S @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    r2 = 1 - (u @ u) / ((yv - yv.mean()) @ (yv - yv.mean()))
    out = pd.DataFrame({"coef": beta, "hac_se": se, "t": t, "p_value": p,
                        "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se}, index=names)
    out.attrs["r2"] = r2
    out.attrs["n"] = n
    return out
