"""
analyze_waves_bias.py

Analyze Waves 1–7 for potential patterns and biases in supervisor feedback
by gender, race, and age, both per-wave and across waves.

Assumptions:
- CSV files: Wave1Unbiased.csv, Wave2Unbiased.csv, ..., Wave7Unbiased.csv
  in the same directory.
- Columns include at least:
    - 'Supervisor'
    - 'Supervisor Feedback'
    - 'Gender'
    - 'Race'
    - 'Age'
    - Several numeric rating columns (1–5 scale, etc.)

Usage:
    python analyze_waves_bias.py
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path


# --- Simple lexicon-based sentiment helper (no external libraries) -----------

POS_WORDS = {
    "excellent", "great", "strong", "reliable", "positive", "improved",
    "outstanding", "helpful", "supportive", "consistent", "proactive",
    "effective", "safe", "thorough", "professional"
}

NEG_WORDS = {
    "poor", "weak", "inconsistent", "problem", "issue", "concern",
    "struggle", "difficulty", "late", "negative", "risky", "careless",
    "unsafe", "unreliable", "needs improvement", "lacking"
}


def simple_sentiment_score(text: str) -> float:
    """Very simple sentiment score: (pos - neg) / token_count."""
    if not isinstance(text, str) or not text.strip():
        return 0.0

    text_lower = text.lower()
    tokens = text_lower.split()
    if not tokens:
        return 0.0

    pos = sum(1 for w in tokens if w in POS_WORDS)
    neg = sum(1 for w in tokens if w in NEG_WORDS)
    return (pos - neg) / len(tokens)


# --- Core helpers -----------------------------------------------------------

def detect_rating_columns(df: pd.DataFrame) -> list:
    """Heuristically detect numeric rating columns (exclude IDs, Age-like cols)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_keywords = ["id", "zip", "age"]
    rating_cols = [
        c for c in numeric_cols
        if not any(kw.lower() in c.lower() for kw in exclude_keywords)
    ]
    return rating_cols


def summarize_group_stats(group: pd.DataFrame, rating_cols: list) -> dict:
    """Compute overall rating mean and sentiment mean for a group of rows."""
    stats = {}
    if rating_cols and not group.empty:
        stats["rating_mean"] = group[rating_cols].mean().mean()
    else:
        stats["rating_mean"] = np.nan

    if "Supervisor Feedback" in group.columns:
        sentiments = group["Supervisor Feedback"].dropna().apply(simple_sentiment_score)
        stats["sentiment_mean"] = sentiments.mean() if not sentiments.empty else np.nan
    else:
        stats["sentiment_mean"] = np.nan

    stats["n"] = len(group)
    return stats


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Create a coarse age band column for analysis."""
    if "Age" not in df.columns:
        return df

    def _band(age):
        try:
            a = float(age)
        except Exception:
            return np.nan
        if a < 30:
            return "<30"
        elif a < 45:
            return "30–44"
        elif a < 60:
            return "45–59"
        else:
            return "60+"

    df = df.copy()
    df["AgeBand"] = df["Age"].apply(_band)
    return df


# --- Wave-level bias summaries ---------------------------------------------

def print_wave_dim_bias(df_wave, wave_label, dim_name, rating_cols,
                        min_n=5, rating_diff_threshold=0.3, sentiment_diff_threshold=0.01):
    """
    For one wave and one demographic dimension (Gender/Race/AgeBand),
    print stats and flag potential bias.
    """
    if dim_name not in df_wave.columns:
        return

    values = df_wave[dim_name].dropna().unique()
    if len(values) < 2:
        return

    print(f"\n  [{wave_label}] {dim_name} patterns:")
    stats_by_value = {}

    for v in values:
        g = df_wave[df_wave[dim_name] == v]
        stats_by_value[v] = summarize_group_stats(g, rating_cols)

    # Print basic stats
    for v, stats in stats_by_value.items():
        print(
            f"    {dim_name} = {v!r}: "
            f"n={stats['n']}, "
            f"mean rating={stats['rating_mean']:.2f} "
            f"(simple sentiment={stats['sentiment_mean']:.3f})"
        )

    # Pairwise comparisons for bias
    for v1, v2 in combinations(values, 2):
        s1 = stats_by_value[v1]
        s2 = stats_by_value[v2]

        # Need enough data overall to say something meaningful
        if s1["n"] < min_n or s2["n"] < min_n:
            continue

        r1, r2 = s1["rating_mean"], s2["rating_mean"]
        if not np.isnan(r1) and not np.isnan(r2):
            diff = r1 - r2
            if abs(diff) >= rating_diff_threshold:
                favored = v1 if diff > 0 else v2
                under = v2 if diff > 0 else v1
                print(
                    f"    ⚠ Potential rating bias: in {wave_label}, "
                    f"{favored!r} {dim_name.lower()} group is rated higher than {under!r} "
                    f"by {abs(diff):.2f} points (n={s1['n']} vs n={s2['n']})."
                )

        m1, m2 = s1["sentiment_mean"], s2["sentiment_mean"]
        if not np.isnan(m1) and not np.isnan(m2):
            sdiff = m1 - m2
            if abs(sdiff) >= sentiment_diff_threshold:
                favored = v1 if sdiff > 0 else v2
                under = v2 if sdiff > 0 else v1
                print(
                    f"    ⚠ Potential tone bias: in {wave_label}, "
                    f"feedback for {favored!r} {dim_name.lower()} group is more positive than "
                    f"{under!r} by {abs(sdiff):.3f} (simple sentiment)."
                )


def summarize_bias_by_wave(df_all, rating_cols):
    """Summarize demographic bias separately for each wave (across all supervisors)."""
    print("\n" + "=" * 80)
    print("OVERALL DEMOGRAPHIC PATTERNS BY WAVE (ALL SUPERVISORS)")
    print("=" * 80)

    for wave in sorted(df_all["Wave"].unique()):
        df_wave = df_all[df_all["Wave"] == wave]
        wave_label = f"Wave {wave}"

        overall = summarize_group_stats(df_wave, rating_cols)
        print("\n" + "-" * 80)
        print(f"{wave_label}: n={len(df_wave)} employees")
        print(
            f"  Overall mean rating: {overall['rating_mean']:.2f} "
            f"(simple sentiment={overall['sentiment_mean']:.3f})"
        )

        for dim in ["Gender", "Race", "AgeBand"]:
            print_wave_dim_bias(df_wave, wave_label, dim, rating_cols)


# --- Supervisor-level cross-wave bias --------------------------------------

def summarize_supervisor_bias_over_time(df_all, rating_cols,
                                        min_n_per_wave=2,
                                        min_waves=2,
                                        rating_diff_threshold=0.3):
    """
    For each supervisor, look across waves for consistent patterns:
    e.g., consistently rating men higher than women, or one racial group lower.
    """
    print("\n" + "=" * 80)
    print("SUPERVISOR-LEVEL CROSS-WAVE BIAS PATTERNS")
    print("=" * 80)

    supers = sorted(df_all["Supervisor"].dropna().unique())

    for sup in supers:
        df_sup = df_all[df_all["Supervisor"] == sup]
        if df_sup.empty:
            continue

        print("\n" + "-" * 80)
        print(f"Supervisor: {sup}")
        print("-" * 80)

        overall = summarize_group_stats(df_sup, rating_cols)
        print(
            f"  Overall mean rating across all waves: {overall['rating_mean']:.2f} "
            f"(simple sentiment={overall['sentiment_mean']:.3f})"
        )

        for dim in ["Gender", "Race", "AgeBand"]:
            if dim not in df_sup.columns:
                continue

            values = df_sup[dim].dropna().unique()
            if len(values) < 2:
                continue

            print(f"\n  {dim} patterns across waves:")

            for v1, v2 in combinations(values, 2):
                diffs = []
                waves_used = []

                for wave in sorted(df_sup["Wave"].unique()):
                    w_subset = df_sup[df_sup["Wave"] == wave]
                    g1 = w_subset[w_subset[dim] == v1]
                    g2 = w_subset[w_subset[dim] == v2]

                    if len(g1) < min_n_per_wave or len(g2) < min_n_per_wave:
                        continue

                    s1 = summarize_group_stats(g1, rating_cols)
                    s2 = summarize_group_stats(g2, rating_cols)
                    if np.isnan(s1["rating_mean"]) or np.isnan(s2["rating_mean"]):
                        continue

                    diff = s1["rating_mean"] - s2["rating_mean"]
                    diffs.append(diff)
                    waves_used.append(wave)

                if len(diffs) >= min_waves:
                    mean_diff = float(np.mean(diffs))
                    if abs(mean_diff) >= rating_diff_threshold:
                        favored = v1 if mean_diff > 0 else v2
                        under = v2 if mean_diff > 0 else v1
                        print(
                            f"    ⚠ Potential consistent rating bias: "
                            f"across {len(waves_used)} waves (Wave(s) {waves_used}), "
                            f"{favored!r} {dim.lower()} group is rated higher than {under!r} "
                            f"by an average of {abs(mean_diff):.2f} points."
                        )
                    else:
                        print(
                            f"    No strong consistent rating gap between {v1!r} and {v2!r} "
                            f"across waves (avg diff={mean_diff:.2f})."
                        )


# --- Combined narrative summaries ------------------------------------------

def generate_overall_demographic_conclusions(df_all, rating_cols):
    """
    High-level narrative about demographic patterns across all waves.
    This is more 'executive summary' style.
    """
    parts = []

    # Overall mean rating
    overall = summarize_group_stats(df_all, rating_cols)
    parts.append(
        f"Across all waves, the overall average rating is {overall['rating_mean']:.2f} "
        f"with an average sentiment score of {overall['sentiment_mean']:.3f} in written feedback."
    )

    for dim in ["Gender", "Race", "AgeBand"]:
        if dim not in df_all.columns:
            continue

        values = df_all[dim].dropna().unique()
        if len(values) < 2:
            continue

        # Compute average rating per group across all waves
        group_means = {}
        for v in values:
            g = df_all[df_all[dim] == v]
            s = summarize_group_stats(g, rating_cols)
            group_means[v] = s["rating_mean"]

        # Find largest pairwise gap
        best_pair = None
        best_gap = 0.0
        for v1, v2 in combinations(values, 2):
            m1, m2 = group_means[v1], group_means[v2]
            if np.isnan(m1) or np.isnan(m2):
                continue
            gap = abs(m1 - m2)
            if gap > best_gap:
                best_gap = gap
                best_pair = (v1, v2, m1, m2)

        if best_pair and best_gap >= 0.3:
            v1, v2, m1, m2 = best_pair
            if m1 > m2:
                favored, under, diff = v1, v2, m1 - m2
            else:
                favored, under, diff = v2, v1, m2 - m1

            parts.append(
                f"For {dim.lower()}, {favored!r} employees receive higher ratings on average "
                f"than {under!r} employees by about {diff:.2f} points across all waves."
            )
        else:
            parts.append(
                f"For {dim.lower()}, no large sustained rating gaps appear across all waves."
            )

    return " ".join(parts)


# --- Main orchestration -----------------------------------------------------

def main():
    # Adjust here if you change file naming
    wave_files = [f"Wave{i}Unbiased.csv" for i in range(1, 8)]

    dfs = []
    for i, fname in enumerate(wave_files, start=1):
        path = Path(fname)
        if not path.exists():
            print(f"Warning: {fname} not found, skipping this wave.")
            continue

        df = pd.read_csv(path)

        # Normalize column names (fix trailing spaces etc.)
        df.columns = df.columns.str.strip()

        # Add wave indicator
        df["Wave"] = i

        dfs.append(df)

    if not dfs:
        print("No wave files loaded. Check file names and paths.")
        return

    # Concatenate all waves
    df_all = pd.concat(dfs, ignore_index=True)

    # Add AgeBand and detect rating columns
    df_all = add_age_band(df_all)
    rating_cols = detect_rating_columns(df_all)

    print("Detected rating columns:", rating_cols)

    # High-level narrative summary
    print("\n" + "=" * 80)
    print("OVERALL DEMOGRAPHIC CONCLUSIONS ACROSS ALL WAVES")
    print("=" * 80)
    print(generate_overall_demographic_conclusions(df_all, rating_cols))

    # Wave-by-wave patterns (all supervisors)
    summarize_bias_by_wave(df_all, rating_cols)

    # Supervisor-level cross-wave patterns
    summarize_supervisor_bias_over_time(df_all, rating_cols)


if __name__ == "__main__":
    main()
