# P.Bazanov, Improved & Modularized for Clarity

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Optional

# =========================
# --- DATA LOADING ---
# =========================

def load_data(ben_path: str, claims_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ben = pd.read_csv(ben_path, dtype={'DESYNPUF_ID': str})
    claims = pd.read_csv(claims_path, dtype={'DESYNPUF_ID': str})
    return ben, claims

# =========================
# --- DATA PROCESSING ---
# =========================

CHRONIC_COLS = [
    'SP_ALZHDMTA','SP_CHF','SP_CHRNKIDN','SP_CNCR','SP_COPD',
    'SP_DEPRESSN','SP_DIABETES','SP_ISCHMCHT','SP_OSTEOPRS',
    'SP_RA_OA','SP_STRKETIA'
]
CHRONIC_MAP = {
    'SP_ALZHDMTA':'Alzheimer','SP_CHF':'Heart Failure','SP_CHRNKIDN':'Chronic Kidney Disease',
    'SP_CNCR':'Cancer','SP_COPD':'COPD','SP_DEPRESSN':'Depression','SP_DIABETES':'Diabetes',
    'SP_ISCHMCHT':'Ischemic Heart Disease','SP_OSTEOPRS':'Osteoporosis','SP_RA_OA':'RA/OA',
    'SP_STRKETIA':'Stroke/TIA'
}

def make_chronic_combo(row: pd.Series) -> str:
    positives = [CHRONIC_MAP[c] for c in CHRONIC_COLS if row.get(c, 0) == 1]
    if len(positives) >= 3:
        return 'Multiple'
    elif not positives:
        return 'None'
    return ', '.join(sorted(positives))

def process_beneficiary(ben: pd.DataFrame) -> pd.DataFrame:
    ben = ben.copy()
    ben['chronic_combo'] = ben.apply(make_chronic_combo, axis=1)
    return ben

def merge_data(ben: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        ben[['DESYNPUF_ID', 'chronic_combo', 'BENE_RACE_CD']],
        claims,
        on='DESYNPUF_ID',
        how='inner'
    )

# =========================
# --- PLOTTING UTILITIES ---
# =========================

def save_plot(fig_name: str):
    plt.tight_layout()
    plt.savefig(fig_name, dpi=200)
    plt.close()

def add_bar_labels(ax, values):
    for i, v in enumerate(values):
        ax.text(v + max(values)*0.01, i, f"{int(v):,}", va='center', fontsize=11, fontweight='bold')

# =========================
# --- ANALYTICS FUNCTIONS ---
# =========================

def plot_race_distribution(merged: pd.DataFrame, out: str = 'race_distribution.png'):
    race_counts = merged['BENE_RACE_CD'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        y=race_counts.index.astype(str),
        x=race_counts.values,
        ax=ax,
        palette='viridis'
    )
    plt.title('Race Distribution')
    plt.xlabel('Number of Members')
    plt.ylabel('Race Code')
    add_bar_labels(ax, race_counts.values)
    save_plot(out)

def plot_top_chronic_combos(merged: pd.DataFrame, top_n: int = 10, out: str = 'top10_chronic_combos.png'):
    combo_counts = merged['chronic_combo'].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        y=combo_counts.index,
        x=combo_counts.values,
        ax=ax,
        palette="crest"
    )
    plt.title('Top 10 Most Common Chronic Illness Combinations')
    plt.xlabel('Number of Members')
    plt.ylabel('Chronic Illness Combination')
    add_bar_labels(ax, combo_counts.values)
    save_plot(out)

def plot_highest_total_cost_by_combo(merged: pd.DataFrame, top_n: int = 10, out: str = 'top10_chronic_combos_total_cost.png'):
    total_costs = merged.groupby('chronic_combo')['CLM_PMT_AMT'].sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        y=total_costs.index,
        x=total_costs.values,
        ax=ax,
        palette="flare"
    )
    plt.title('Top 10 Chronic Illness Combos by Total Cost')
    plt.xlabel('Total Cost')
    plt.ylabel('Chronic Illness Combination')
    add_bar_labels(ax, total_costs.values)
    save_plot(out)
def plot_top_chronic_combos_percentile(merged, percentile=75):
    # Filter out 'Multiple' and 'None'
    filtered = merged[~merged['chronic_combo'].isin(['Multiple', 'None'])]
    # Count combos
    combo_counts = filtered['chronic_combo'].value_counts()
    # Get cutoff for percentile
    cutoff = np.percentile(combo_counts.values, percentile)
    # Select only combos above cutoff (>= in case there are ties)
    top_combos = combo_counts[combo_counts >= cutoff]
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        y=top_combos.index,
        x=top_combos.values,
        hue=top_combos.index,
        palette="crest",
        legend=False,
        dodge=False
    )
    plt.title(f'Chronic Illness Combos (above {percentile}th percentile by count)')
    plt.xlabel('Number of Members')
    plt.ylabel('Chronic Illness Combination')
    for i, v in enumerate(top_combos.values):
        ax.text(v + max(top_combos.values)*0.01, i, f"{int(v):,}", va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("top_chronic_combos_percentile.png")
    plt.close()
def provider_stats_filtered(provider_stats, merged):
    counts = (merged
              .groupby(['AT_PHYSN_NPI', 'chronic_combo'])['DESYNPUF_ID']
              .nunique()
              .reset_index(name='member_count'))
    to_keep = counts[counts['member_count'] > 1][['AT_PHYSN_NPI', 'chronic_combo']]
    filtered = provider_stats.merge(to_keep, on=['AT_PHYSN_NPI', 'chronic_combo'])
    plt.figure(figsize=(8, 5))
    sns.histplot(filtered['cost_per_member'], bins=50, kde=True, color='orange')
    plt.title('Distribution of Provider Cost Per Member (Filtered: >1 Member)')
    plt.xlabel('Cost Per Member')
    plt.ylabel('Provider-Combo Count')
    plt.tight_layout()
    plt.savefig('provider_cost_per_member_hist_filtered.png')
    plt.close()
    return filtered

def expensive_providers_across_combos(provider_stats, top_percentile=0.75):
    expensive_flags = []
    for combo, grp in provider_stats.groupby('chronic_combo'):
        threshold = grp['cost_per_member'].quantile(top_percentile)
        expensive_flags.append(
            grp.assign(expensive=grp['cost_per_member'] > threshold)
        )
    flagged = pd.concat(expensive_flags)
    summary = (flagged.groupby('AT_PHYSN_NPI')['expensive']
               .sum()
               .sort_values(ascending=False)
               .reset_index(name='num_expensive_combos'))
    top10 = summary.head(10)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=top10,
        x='AT_PHYSN_NPI',
        y='num_expensive_combos',
        hue='AT_PHYSN_NPI',
        palette='Reds_r',
        dodge=False,
        legend=False
    )
    for p in ax.patches:
        ax.annotate(int(p.get_height()),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    plt.title('Top 10 Providers (Most Expensive Across Chronic Combos)')
    plt.xlabel('Provider NPI')
    plt.ylabel('Number of Expensive Chronic Combos')
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig('top10_expensive_providers.png')
    plt.close()
    return summary
def provider_cost_per_member(merged):
    df = (merged
          .groupby(['AT_PHYSN_NPI', 'chronic_combo', 'DESYNPUF_ID'])['CLM_PMT_AMT']
          .sum()
          .reset_index(name='member_cost'))
    provider_stats = (df
        .groupby(['AT_PHYSN_NPI', 'chronic_combo'])['member_cost']
        .mean()
        .reset_index(name='cost_per_member'))
    plt.figure(figsize=(8, 5))
    sns.histplot(provider_stats['cost_per_member'], bins=50, kde=True)
    plt.title('Distribution of Provider Cost Per Member (All Combos)')
    plt.xlabel('Cost Per Member')
    plt.ylabel('Provider-Combo Count')
    plt.tight_layout()
    plt.savefig('provider_cost_per_member_hist.png')
    plt.close()
    return provider_stats

def plot_highest_avg_cost_per_member(merged, top_n=10):
    member_combo_cost = merged.groupby(['DESYNPUF_ID', 'chronic_combo'])['CLM_PMT_AMT'].sum().reset_index()
    avg_per_member = member_combo_cost.groupby('chronic_combo')['CLM_PMT_AMT'].mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        y=avg_per_member.index,
        x=avg_per_member.values,
        hue=avg_per_member.index,
        palette="mako",
        legend=False,
        dodge=False
    )

    plt.title('Top 10 Chronic Illness Combos by Avg Cost per Member')
    plt.xlabel('Average Cost per Member')
    plt.ylabel('Chronic Illness Combination')
    for i, v in enumerate(avg_per_member.values):
        ax.text(v + max(avg_per_member.values)*0.01, i, f"${int(v):,}", va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("top10_chronic_combos_avg_cost.png")
    plt.close()

def plot_highest_total_cost_by_combo_percentile(merged, percentile=99):
    # Exclude 'Multiple' and 'None'
    filtered = merged[~merged['chronic_combo'].isin(['Multiple', 'None'])]
    total_costs = (
        filtered.groupby('chronic_combo')['CLM_PMT_AMT']
        .sum()
        .sort_values(ascending=False)
    )
    # Compute cutoff for 95th percentile
    cutoff = np.percentile(total_costs.values, percentile)
    # Select combos above or equal to cutoff
    top_cost_combos = total_costs[total_costs >= cutoff]
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        y=top_cost_combos.index,
        x=top_cost_combos.values,
        hue=top_cost_combos.index,
        palette="flare",
        legend=False,
        dodge=False
    )
    plt.title(f'Chronic Illness Combos (above {percentile}th percentile by Total Cost)')
    plt.xlabel('Total Cost')
    plt.ylabel('Chronic Illness Combination')
    for i, v in enumerate(top_cost_combos.values):
        ax.text(v + max(top_cost_combos.values)*0.01, i, f"${int(v):,}", va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("top_chronic_combos_total_cost_percentile.png")
    plt.close()

# ... (your imports and previous code here)

def provider_chronic_cost_by_mode(merged, mode='filtered'):
    df = (
        merged
        .groupby(['AT_PHYSN_NPI', 'chronic_combo', 'DESYNPUF_ID'])['CLM_PMT_AMT']
        .sum()
        .reset_index(name='member_total_cost')
    )
    if mode == 'filtered':
        counts = (
            df
            .groupby(['AT_PHYSN_NPI', 'chronic_combo'])['DESYNPUF_ID']
            .nunique()
            .reset_index(name='n_members')
        )
        keep = counts[counts['n_members'] > 1]
        df = pd.merge(
            df,
            keep[['AT_PHYSN_NPI', 'chronic_combo']],
            on=['AT_PHYSN_NPI', 'chronic_combo'],
            how='inner'
        )
    summary = (
        df
        .groupby(['AT_PHYSN_NPI', 'chronic_combo'])
        .agg(
            cost_per_member=('member_total_cost', 'mean'),
            n_members=('DESYNPUF_ID', 'nunique')
        )
        .reset_index()
    )
    return summary

def visualize_provider_chronic_costs(summary, mode_label='standard', top_n=10):
    top_combos = summary['chronic_combo'].value_counts().head(top_n).index
    plot_data = summary[summary['chronic_combo'].isin(top_combos)]
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(
        data=plot_data,
        x='cost_per_member',
        y='chronic_combo',
        order=top_combos,
        palette="vlag"
    )
    plt.title(f'Provider Cost per Member Distribution (Top {top_n} Combos, Mode: {mode_label})')
    plt.xlabel('Cost per Member')
    plt.ylabel('Chronic Illness Combination')
    plt.tight_layout()
    plt.savefig(f'provider_cost_per_member_boxplot_{mode_label}.png')
    plt.close()

def visualize_provider_chronic_costs_percentile(summary, percentile=75, mode_label='standard'):
    filtered = summary[~summary['chronic_combo'].isin(['Multiple', 'None'])]
    combo_counts = filtered['chronic_combo'].value_counts()
    cutoff = np.percentile(combo_counts.values, percentile)
    top_combos = combo_counts[combo_counts >= cutoff].index
    plot_data = filtered[filtered['chronic_combo'].isin(top_combos)]
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(
        data=plot_data,
        x='cost_per_member',
        y='chronic_combo',
        order=top_combos,
        palette="vlag"
    )
    plt.title(f'Provider Cost per Member (Combos above {percentile}th percentile, Mode: {mode_label})')
    plt.xlabel('Cost per Member')
    plt.ylabel('Chronic Illness Combination')
    plt.tight_layout()
    plt.savefig(f'provider_cost_per_member_boxplot_percentile_p{percentile}_{mode_label}.png')
    plt.close()

def plot_cost_distribution_per_chronic(summary, top_n=10, exclude_multiple_none=True):
    data = summary.copy()
    if exclude_multiple_none:
        data = data[~data['chronic_combo'].isin(['Multiple', 'None'])]
    top_combos = data['chronic_combo'].value_counts().head(top_n).index
    plot_data = data[data['chronic_combo'].isin(top_combos)]
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(
        data=plot_data,
        x='cost_per_member',
        y='chronic_combo',
        order=top_combos,
        palette="rocket"
    )
    plt.title(f'Distribution of Provider Cost per Member (Top {top_n} Chronic Illness Combos)')
    plt.xlabel('Cost per Member')
    plt.ylabel('Chronic Illness Combination')
    plt.tight_layout()
    plt.savefig(f'cost_distribution_per_chronic_top{top_n}_{"noMultiNone" if exclude_multiple_none else "all"}.png')
    plt.close()

def consistently_expensive_providers(
    merged, provider_col, illness_col, cost_col, quantile=0.75, min_illnesses=2
):
    # Find, for each chronic illness, providers above the cost quantile
    out = []
    for illness, group in merged.groupby(illness_col):
        if len(group) < 2:
            continue
        q = group[cost_col].quantile(quantile)
        expensive = group[group[cost_col] > q]
        out.append(expensive[[provider_col, illness_col]])
    expensive_providers = pd.concat(out)
    # Count for each provider how many illnesses they are expensive in
    count = expensive_providers.groupby(provider_col)[illness_col].nunique().reset_index()
    count.columns = [provider_col, 'num_expensive_illnesses']
    count = count[count['num_expensive_illnesses'] >= min_illnesses]
    return count
def expensive_providers_across_combos(provider_stats, top_percentile=0.75):
    expensive_flags = []
    for combo, grp in provider_stats.groupby('chronic_combo'):
        threshold = grp['cost_per_member'].quantile(top_percentile)
        expensive_flags.append(
            grp.assign(expensive=grp['cost_per_member'] > threshold)
        )
    flagged = pd.concat(expensive_flags)
    summary = (flagged.groupby('AT_PHYSN_NPI')['expensive']
               .sum()
               .sort_values(ascending=False)
               .reset_index(name='num_expensive_combos'))
    top10 = summary.head(10)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=top10,
        x='AT_PHYSN_NPI',
        y='num_expensive_combos',
        hue='AT_PHYSN_NPI',
        palette='Reds_r',
        dodge=False,
        legend=False
    )
    for p in ax.patches:
        ax.annotate(int(p.get_height()),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    plt.title('Top 10 Providers (Most Expensive Across Chronic Combos)')
    plt.xlabel('Provider NPI')
    plt.ylabel('Number of Expensive Chronic Combos')
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig('top10_expensive_providers.png')
    plt.close()
    return summary
def expensive_providers_across_combos_perc(provider_stats, top_percentile=0.99, top_n=10):
    # Exclude 'Multiple' and 'None'
    filtered = provider_stats[~provider_stats['chronic_combo'].isin(['Multiple', 'None'])]
    expensive_flags = []
    for combo, grp in filtered.groupby('chronic_combo'):
        threshold = grp['cost_per_member'].quantile(top_percentile)
        expensive_flags.append(
            grp.assign(expensive=grp['cost_per_member'] > threshold)
        )
    flagged = pd.concat(expensive_flags)
    summary = (
        flagged.groupby('AT_PHYSN_NPI')['expensive']
        .sum()
        .sort_values(ascending=False)
        .reset_index(name='num_expensive_combos')
    )
    top_providers = summary.head(top_n)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=top_providers,
        x='AT_PHYSN_NPI',
        y='num_expensive_combos',
        hue='AT_PHYSN_NPI',
        palette='Reds_r',
        dodge=False,
        legend=False
    )
    for p in ax.patches:
        ax.annotate(int(p.get_height()),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    plt.title(f'Top {top_n} Providers (Most Expensive Across Chronic Combos, >{int(top_percentile*100)}th Percentile)')
    plt.xlabel('Provider NPI')
    plt.ylabel('Number of Expensive Chronic Combos')
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig('top10_expensive_providers.png')
    plt.close()
    return summary
def plot_expensive_providers(df, provider_col, count_col, top_n=10, filename='expensive_providers.png'):
    top = df.sort_values(count_col, ascending=False).head(top_n)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=top,
        x=provider_col,
        y=count_col,
        palette='Reds_r'
    )
    for i, v in enumerate(top[count_col].values):
        ax.text(i, v + 0.5, f"{int(v)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.title('Top Expensive Providers Across Chronic Illnesses')
    plt.xlabel('Provider NPI')
    plt.ylabel('Number of Expensive Chronic Combos')
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_cost_percentile(merged, illness_col, illness_value, cost_col, quantile=0.75, filename='cost_percentile.png'):
    filtered = merged[merged[illness_col] == illness_value]
    if filtered.empty:
        print(f"No data for {illness_value}")
        return
    plt.figure(figsize=(10, 5))
    ax = sns.histplot(filtered[cost_col], bins=40, kde=True, color='b', alpha=0.5)
    perc = filtered[cost_col].quantile(quantile)
    plt.axvline(perc, color='red', linestyle='--', label=f'{int(quantile*100)}th percentile = ${int(perc):,}')
    plt.legend()
    plt.title(f'Cost Distribution for {illness_value}')
    plt.xlabel('Claim Cost')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Now your main() pipeline is fully supported!

# =========================
# --- MAIN ANALYSIS PIPELINE ---
# =========================

def main():
    # Data paths
    ben_path = 'd:/code/uhg/data/DE1_0_2009_Beneficiary_Summary_File_Sample_20.csv'
    claims_path = 'd:/code/uhg/data/DE1_0_2008_to_2010_Outpatient_Claims_Sample_20.csv'

    # Load & process
    ben, claims = load_data(ben_path, claims_path)
    ben = process_beneficiary(ben)
    merged = merge_data(ben, claims)
    merged.to_csv('d:/code/uhg/data/merged_beneficiary_claims_with_duplicates.csv', index=False)



    # === 1. Plot race distribution (barplot by race code)
    plot_race_distribution(merged)

    # === 2. Show top 10 most common chronic illness combinations (barplot)
    plot_top_chronic_combos(merged)

    # === 3. Show most common chronic combos above the 75th percentile (barplot)
    plot_top_chronic_combos_percentile(merged, percentile=75)

    # === 4. Show chronic combos with highest total claim cost (barplot)
    plot_highest_total_cost_by_combo(merged)

    # === 5. Show combos with highest total claim cost (above 75th percentile, barplot)
    plot_highest_total_cost_by_combo_percentile(merged, percentile=75)

    # === 6. Show chronic combos with highest average cost per member (barplot)
    plot_highest_avg_cost_per_member(merged)

    # === 7. Show histogram: distribution of provider cost per member (all provider-combo pairs)
    provider_stats = provider_cost_per_member(merged)

    # === 8. Show histogram: provider cost per member (excluding single-member combos)
    provider_stats_multi = provider_stats_filtered(provider_stats, merged)

    # === 9. Show number of expensive chronic combos for each provider (barplot, top 10)
    expensive_prov_summary = expensive_providers_across_combos(provider_stats_multi)
    print(expensive_prov_summary.head(10))  # Review the top expensive providers

    # === 10. Show provider cost per member distribution for all combos (boxplot)
    summary_standard = provider_chronic_cost_by_mode(merged, 'standard')
    visualize_provider_chronic_costs(summary_standard, mode_label='standard')

    # === 11. Boxplot by chronic combo, combos above 75th percentile (provider cost per member)
    visualize_provider_chronic_costs_percentile(summary_standard, percentile=75, mode_label='standard')

    # === 12. Boxplot: filtered provider/chronic combos (removing single-member combos)
    summary_filtered = provider_chronic_cost_by_mode(merged, 'filtered')
    visualize_provider_chronic_costs(summary_filtered, mode_label='filtered')

    # === 13. Boxplot: filtered combos, only top by count (above 75th percentile)
    visualize_provider_chronic_costs_percentile(summary_filtered, percentile=75, mode_label='filtered')

    # === 14. Boxplot: provider cost distribution per chronic combo (excluding 'Multiple'/'None')
    plot_cost_distribution_per_chronic(summary_filtered, top_n=10, exclude_multiple_none=True)

    # === 15. Boxplot: provider cost distribution per chronic combo (including 'Multiple'/'None')
    plot_cost_distribution_per_chronic(summary_filtered, top_n=10, exclude_multiple_none=False)

    # === 16. List consistently expensive providers across illnesses (print top 20 DataFrame)
    result = consistently_expensive_providers(
        merged, 'AT_PHYSN_NPI', 'chronic_combo', 'CLM_PMT_AMT', quantile=0.75, min_illnesses=2
    )
    result_sorted = result.sort_values('num_expensive_illnesses', ascending=False).head(20)
    print(result_sorted)

    # === 17. Barplot: top 10 consistently expensive providers
    plot_expensive_providers(result_sorted, provider_col='AT_PHYSN_NPI',
                             count_col='num_expensive_illnesses', top_n=10,
                             filename='expensive_providers.png')

    # === 18. Histogram: claim cost distribution for "Diabetes" (red line = 75th percentile)
    plot_cost_percentile(merged, illness_col='chronic_combo', illness_value='Diabetes',
                         cost_col='CLM_PMT_AMT', quantile=0.75, filename='diabetes_cost_percentile.png')

    # === 19. Histogram: claim cost distribution for "Diabetes, Ischemic Heart Disease"
    plot_cost_percentile(merged, illness_col='chronic_combo', illness_value='Diabetes, Ischemic Heart Disease',
                         cost_col='CLM_PMT_AMT', quantile=0.75, filename='diabetes_Ischemic Heart_cost_percentile.png')

    # === 20. Histogram: claim cost distribution for "Ischemic Heart Disease"
    plot_cost_percentile(merged, illness_col='chronic_combo', illness_value='Ischemic Heart Disease',
                         cost_col='CLM_PMT_AMT', quantile=0.75, filename='Ischemic Heart_diabetes_cost_percentile.png')

    # === 21. (Optional) Print available columns for quick data schema check
    print(merged.columns.tolist())
    print(merged.head())


if __name__ == "__main__":
    main()
