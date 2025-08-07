#Top priority clusters
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

def safe_print(df, columns):
    """Print only the columns that exist in the DataFrame, warn about missing ones."""
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: columns missing in DataFrame and will be skipped: {missing_cols}")
    print(df[existing_cols])

def plot_top_anomaly_clusters(tsne_df, top_n=10, savefig=True, filename='tsne_top_anomaly_clusters.png', xlim=(-150, 150), ylim=(-150, 150)):
    """
    Detects and plots the top anomaly clusters using t-SNE and IsolationForest.
    Highlights top anomalies in red, labels with cluster names.
    """
    required_cols = {'TSNE1', 'TSNE2', 'ChronicCombo', 'avg_cost'}
    if not required_cols.issubset(tsne_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    clf = IsolationForest(random_state=42, contamination=0.02)
    tsne_df = tsne_df.copy()
    tsne_df['anomaly_score'] = -clf.fit(tsne_df[['TSNE1', 'TSNE2']]).score_samples(tsne_df[['TSNE1', 'TSNE2']])
    top_anomalies = tsne_df.nlargest(top_n, 'anomaly_score')

    plt.figure(figsize=(13, 9))
    ax = sns.scatterplot(
        data=tsne_df, x='TSNE1', y='TSNE2',
        hue='ChronicCombo', size='avg_cost',
        sizes=(20, 120), alpha=0.3, palette='tab10', legend='brief'
    )
    ax.scatter(
        top_anomalies['TSNE1'], top_anomalies['TSNE2'],
        color='red', s=140, marker='o', label='Top Anomalies', edgecolor='black', zorder=5
    )
    for _, row in top_anomalies.iterrows():
        ax.text(row.TSNE1, row.TSNE2, f"{row['ChronicCombo']}", fontsize=10, color='red', fontweight='bold', alpha=0.85)
    ax.set_title(f"Top {top_n} Anomaly Clusters (Red) by t-SNE/IsolationForest")
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2)
    plt.tight_layout()
    if savefig:
        plt.savefig(filename)
    plt.show()

    print("\n=== Top Anomaly Clusters ===")
    safe_print(top_anomalies, ['Provider', 'ChronicCombo', 'avg_cost', 'n_members', 'TSNE1', 'TSNE2'])
    return top_anomalies

def plot_top_priority_large_blobs(tsne_df, top_n=10, savefig=True, filename='tsne_top_large_blobs.png', xlim=(-150, 150), ylim=(-150, 150)):
    """
    Plots the largest average cost clusters (large blobs) in blue, with annotation.
    """
    required_cols = {'TSNE1', 'TSNE2', 'ChronicCombo', 'avg_cost'}
    if not required_cols.issubset(tsne_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    large_blobs = tsne_df.nlargest(top_n, 'avg_cost')

    plt.figure(figsize=(13, 9))
    ax = sns.scatterplot(
        data=tsne_df, x='TSNE1', y='TSNE2',
        hue='ChronicCombo', size='avg_cost',
        sizes=(20, 120), alpha=0.3, palette='tab10', legend='brief'
    )
    ax.scatter(
        large_blobs['TSNE1'], large_blobs['TSNE2'],
        color='blue', s=180, marker='o', label='Top Large Blobs', edgecolor='black', zorder=5
    )
    for _, row in large_blobs.iterrows():
        ax.text(row.TSNE1, row.TSNE2, f"{row['ChronicCombo']}", fontsize=10, color='blue', fontweight='bold', alpha=0.85)
    ax.set_title(f"Top {top_n} Priority Large Cost Blobs (Blue) by t-SNE")
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2)
    plt.tight_layout()
    if savefig:
        plt.savefig(filename)
    plt.show()

    print("\n=== Top Priority Large Blobs ===")
    safe_print(large_blobs, ['Provider', 'ChronicCombo', 'avg_cost', 'n_members', 'TSNE1', 'TSNE2'])
    return large_blobs

# --- Usage Example ---
if __name__ == "__main__":
    tsne_df = pd.read_csv('tsne.csv')
    plot_top_anomaly_clusters(tsne_df, top_n=15)
    plot_top_priority_large_blobs(tsne_df, top_n=15)
