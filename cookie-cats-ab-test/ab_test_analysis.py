"""
A/B test analysis: Cookie Cats, gate at level 30 (control) vs level 40 (treatment).

Question: does moving the gate from level 30 to level 40 change player retention?

Columns of the dataset:
    userid          unique player id
    version         gate_30 (control) or gate_40 (treatment)
    sum_gamerounds  game rounds played in the first 14 days after install
    retention_1     did the player come back 1 day after install (True/False)
    retention_7     did the player come back 7 days after install (True/False)

Plan (decided before looking at the results):
    primary metric    : 7-day retention, alpha = 0.05
    secondary metrics : 1-day retention and game rounds, Bonferroni alpha = 0.05 / 2
    analysis          : all users as assigned (intention to treat)

Все изи 
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files, do not open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cookie_cats.csv"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

CONTROL, TREATMENT = "gate_30", "gate_40"
ALPHA_PRIMARY = 0.05
ALPHA_SECONDARY = 0.05 / 2  # Bonferroni correction for two secondary metrics
N_BOOT = 10_000
SEED = 42

# Chart colors: one fixed color per group, quiet axes
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = {CONTROL: "#2a78d6", TREATMENT: "#eb6834"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 11,
})


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  saved images/{name}")


# ------------------------------------------------------------------ data
def load():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"File not found: {DATA_PATH}\nRun  python get_data.py  first, "
            "or put cookie_cats.csv into the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"userid", "version", "sum_gamerounds", "retention_1", "retention_7"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Columns missing in the file: {sorted(missing)}")
    if set(df["version"].unique()) != {CONTROL, TREATMENT}:
        raise SystemExit(f"Unexpected values in 'version': {df['version'].unique()}")
    for col in ("retention_1", "retention_7"):
        df[col] = df[col].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).astype(int)
    return df


def data_quality(df):
    counts = df["version"].value_counts()
    n1, n2 = counts[CONTROL], counts[TREATMENT]
    # Sample ratio mismatch: with a 50/50 design the group sizes should be almost equal
    srm_p = stats.chisquare([n1, n2]).pvalue
    zero = (df["sum_gamerounds"] == 0).mean()
    q = df["sum_gamerounds"].quantile(0.999)
    info = {
        "rows": len(df),
        "duplicate_userids": int(df["userid"].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "n_control": int(n1),
        "n_treatment": int(n2),
        "sample_ratio_p_value": srm_p,
        "share_zero_rounds": zero,
        "max_rounds": int(df["sum_gamerounds"].max()),
        "p99_9_rounds": float(q),
    }
    print("\n=== Data quality")
    for key, value in info.items():
        print(f"  {key:<22} {value:,.4f}" if isinstance(value, float) else f"  {key:<22} {value:,}")
    return info


# ------------------------------------------------------------------ statistics
def two_proportion_test(df, col):
    a = df.loc[df["version"] == CONTROL, col]
    b = df.loc[df["version"] == TREATMENT, col]
    n1, n2 = len(a), len(b)
    p1, p2 = a.mean(), b.mean()
    diff = p2 - p1
    pooled = (a.sum() + b.sum()) / (n1 + n2)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = diff / se_pooled
    p_value = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    chi2_p = stats.chi2_contingency(pd.crosstab(df["version"], df[col]), correction=False)[1]
    return {
        "metric": col,
        "rate_control": p1,
        "rate_treatment": p2,
        "diff_pp": diff * 100,
        "ci_low_pp": (diff - 1.96 * se) * 100,
        "ci_high_pp": (diff + 1.96 * se) * 100,
        "relative_change_pct": diff / p1 * 100,
        "z": z,
        "p_value": p_value,
        "p_value_chi2_check": chi2_p,
        "n_control": n1,
        "n_treatment": n2,
    }


def rounds_test(df):
    a = df.loc[df["version"] == CONTROL, "sum_gamerounds"]
    b = df.loc[df["version"] == TREATMENT, "sum_gamerounds"]
    cap = df["sum_gamerounds"].quantile(0.999)  # one extreme player can distort the mean
    mw_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    welch_p = stats.ttest_ind(a.clip(upper=cap), b.clip(upper=cap), equal_var=False).pvalue
    return {
        "metric": "sum_gamerounds",
        "median_control": a.median(),
        "median_treatment": b.median(),
        "mean_control_capped": a.clip(upper=cap).mean(),
        "mean_treatment_capped": b.clip(upper=cap).mean(),
        "p_value_mann_whitney": mw_p,
        "p_value_welch_capped": welch_p,
    }


def bootstrap_difference(df, col, rng):
    """Bootstrap of the difference in rates (in percentage points)."""
    a = df.loc[df["version"] == CONTROL, col]
    b = df.loc[df["version"] == TREATMENT, col]
    boot_a = rng.binomial(len(a), a.mean(), N_BOOT) / len(a)
    boot_b = rng.binomial(len(b), b.mean(), N_BOOT) / len(b)
    return (boot_b - boot_a) * 100


def minimum_detectable_effect(n_per_group, base_rate, alpha=0.05, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return (z_alpha + z_power) * np.sqrt(2 * base_rate * (1 - base_rate) / n_per_group)


def sample_size_needed(base_rate, effect, alpha=0.05, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return 2 * (z_alpha + z_power) ** 2 * base_rate * (1 - base_rate) / effect ** 2


# ------------------------------------------------------------------ charts
def chart_retention(df, tests):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    metrics = [("retention_1", "1-day retention"), ("retention_7", "7-day retention")]
    width = 0.34
    for j, group in enumerate((CONTROL, TREATMENT)):
        for i, (col, _) in enumerate(metrics):
            sub = df.loc[df["version"] == group, col]
            ci = stats.binomtest(int(sub.sum()), len(sub)).proportion_ci(method="wilson")
            rate = sub.mean() * 100
            x = i + (j - 0.5) * width
            ax.bar(x, rate, width * 0.92, color=COLORS[group], label=group if i == 0 else None)
            ax.errorbar(x, rate, yerr=[[rate - ci.low * 100], [ci.high * 100 - rate]],
                        color=INK, linewidth=1.2, capsize=4)
            ax.text(x, ci.high * 100 + 1.2, f"{rate:.1f}%", ha="center", color=INK, fontsize=10)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, max(t["rate_control"] for t in tests) * 100 * 1.35)
    ax.set_ylabel("Share of players, %  (bars: 95% confidence interval)")
    ax.set_title("Player retention by gate position", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    save(fig, "01_retention_by_group.png")


def chart_bootstrap(boot, observed, ci_low, ci_high):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    counts, _, _ = ax.hist(boot, bins=60, color=COLORS[CONTROL], alpha=0.85)
    ax.set_ylim(0, counts.max() * 1.55)
    ax.axvline(0, color=INK2, linestyle="--", linewidth=1.5, ymax=0.72)
    for edge in (ci_low, ci_high):
        ax.axvline(edge, color=COLORS[TREATMENT], linewidth=2, ymax=0.72)
    ax.text(0.02, 0.95,
            f"Observed difference: {observed:+.2f} pp\n95% interval: [{ci_low:+.2f}, {ci_high:+.2f}] pp\n"
            "dashed line = no difference",
            transform=ax.transAxes, va="top", color=INK, fontsize=10)
    ax.set_xlabel("Difference in 7-day retention, percentage points (gate_40 minus gate_30)")
    ax.set_ylabel("Bootstrap samples")
    ax.set_title("How uncertain is the 7-day retention difference?", loc="left", fontweight="bold")
    style(ax)
    save(fig, "02_bootstrap_retention_7.png")


def chart_rounds(df):
    positive = df[df["sum_gamerounds"] > 0]
    top = df["sum_gamerounds"].quantile(0.999)
    bins = np.unique(np.floor(np.logspace(0, np.log10(top), 40)).astype(int))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for group in (CONTROL, TREATMENT):
        values = positive.loc[positive["version"] == group, "sum_gamerounds"].clip(upper=top)
        ax.hist(values, bins=bins, histtype="step", linewidth=2.2, density=True,
                color=COLORS[group], label=group)
    ax.set_xscale("log")
    ax.set_xlabel("Game rounds in the first 14 days (log scale, players with 1+ rounds)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of game rounds by group",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False)
    style(ax)
    save(fig, "03_gamerounds_distribution.png")


# ------------------------------------------------------------------ report
def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def verdict(test, alpha):
    if test["p_value"] < alpha:
        direction = "higher" if test["diff_pp"] > 0 else "lower"
        return f"statistically significant ({direction} with gate_40)"
    return "no statistically significant difference"


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    quality = data_quality(df)

    r1 = two_proportion_test(df, "retention_1")
    r7 = two_proportion_test(df, "retention_7")
    rounds = rounds_test(df)
    table = pd.DataFrame([r1, r7])
    table["alpha"] = [ALPHA_SECONDARY, ALPHA_PRIMARY]
    table["role"] = ["secondary", "primary"]
    table["significant"] = table["p_value"] < table["alpha"]
    table.round(5).to_csv(RESULTS_DIR / "retention_tests.csv", index=False)
    pd.DataFrame([rounds]).round(5).to_csv(RESULTS_DIR / "gamerounds_tests.csv", index=False)

    print("\n=== Retention tests (gate_40 minus gate_30)")
    print(table[["metric", "role", "rate_control", "rate_treatment", "diff_pp",
                 "ci_low_pp", "ci_high_pp", "p_value", "significant"]].round(4).to_string(index=False))
    print("\n=== Game rounds")
    for key, value in rounds.items():
        print(f"  {key:<24} {value}")

    boot = bootstrap_difference(df, "retention_7", rng)
    boot_low, boot_high = np.percentile(boot, [2.5, 97.5])

    n_group = min(quality["n_control"], quality["n_treatment"])
    mde = minimum_detectable_effect(n_group, r7["rate_control"]) * 100
    need_1pp = sample_size_needed(r7["rate_control"], 0.01)
    print(f"\n=== Power: with {n_group:,} players per group the smallest reliably detectable "
          f"change in 7-day retention is about {mde:.2f} pp")

    print("\nDrawing charts...")
    chart_retention(df, [r1, r7])
    chart_bootstrap(boot, r7["diff_pp"], boot_low, boot_high)
    chart_rounds(df)

    lines = [
        f"{quality['rows']:,} players: {quality['n_control']:,} in gate_30 and {quality['n_treatment']:,} in gate_40.",
        f"Group sizes {'look balanced' if quality['sample_ratio_p_value'] >= 0.01 else 'differ more than expected for a 50/50 split'} "
        f"(sample ratio check p {fmt_p(quality['sample_ratio_p_value'])}).",
        f"{quality['share_zero_rounds']:.1%} of players installed the game but played zero rounds; "
        f"the maximum is {quality['max_rounds']:,} rounds (extreme values were capped for the mean comparison).",
        f"7-day retention (primary): {r7['rate_control']:.2%} in gate_30 vs {r7['rate_treatment']:.2%} in gate_40, "
        f"difference {r7['diff_pp']:+.2f} pp (95% CI {r7['ci_low_pp']:+.2f} to {r7['ci_high_pp']:+.2f}), "
        f"p {fmt_p(r7['p_value'])}: {verdict(r7, ALPHA_PRIMARY)}.",
        f"1-day retention (secondary): {r1['rate_control']:.2%} vs {r1['rate_treatment']:.2%}, "
        f"difference {r1['diff_pp']:+.2f} pp, p {fmt_p(r1['p_value'])}: {verdict(r1, ALPHA_SECONDARY)}.",
        f"Game rounds (secondary): median {rounds['median_control']:.0f} vs {rounds['median_treatment']:.0f}, "
        f"Mann-Whitney p {fmt_p(rounds['p_value_mann_whitney'])}.",
        f"Bootstrap 95% interval for the 7-day retention difference: {boot_low:+.2f} to {boot_high:+.2f} pp.",
        f"With about {n_group:,} players per group the test can reliably detect a change of {mde:.2f} pp or more in 7-day "
        f"retention; detecting 1 pp would need about {need_1pp:,.0f} players per group.",
    ]
    if r7["p_value"] < ALPHA_PRIMARY and r7["diff_pp"] < 0:
        lines.append("Recommendation: keep the gate at level 30, because gate_40 lowers 7-day retention.")
    elif r7["p_value"] < ALPHA_PRIMARY:
        lines.append("Recommendation: move the gate to level 40, because it increases 7-day retention.")
    else:
        lines.append("Recommendation: the data does not show a reliable effect on 7-day retention; "
                     "decide on other grounds (for example revenue) or run a larger test.")
    text = "\n".join(f"- {line}" for line in lines)
    (RESULTS_DIR / "findings.txt").write_text(text, encoding="utf-8")
    print("\n" + "=" * 70 + "\nKEY FINDINGS (copy into README)\n" + "=" * 70)
    print(text)
    print("\nDone. Tables are in results/, charts are in images/.")


if __name__ == "__main__":
    main()
