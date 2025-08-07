#P.Bazanov
#	Stochastic clustering insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Load merged data
merged = pd.read_csv('/data/merged_beneficiary_claims_with_duplicates.csv')

def tsne_provider_clustering(merged, n_components=2, random_state=42):
    # Step 1: Aggregate provider+chronic_combo features
    group = merged.groupby(['AT_PHYSN_NPI', 'chronic_combo']).agg(
        avg_cost=('CLM_PMT_AMT', 'mean'),
        n_members=('DESYNPUF_ID', 'nunique'),
        total_cost=('CLM_PMT_AMT', 'sum')
    ).reset_index()

    # One-hot encode chronic_combo (robust to sklearn version)
    try:
        enc = OneHotEncoder(sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(sparse=False)
    chronic_ohe = enc.fit_transform(group[['chronic_combo']])
    chronic_cols = enc.get_feature_names_out(['chronic_combo'])
    X = np.concatenate([
        group[['avg_cost', 'n_members', 'total_cost']].values,
        chronic_ohe
    ], axis=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE embedding
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
    X_embedded = tsne.fit_transform(X_scaled)

    # DataFrame for plotting
    tsne_df = pd.DataFrame(X_embedded, columns=['TSNE1', 'TSNE2'])
    tsne_df['Provider'] = group['AT_PHYSN_NPI'].astype(str)
    tsne_df['ChronicCombo'] = group['chronic_combo']
    tsne_df['avg_cost'] = group['avg_cost']

    return tsne_df

def visualize_tsne_clusters(tsne_df, color_by='ChronicCombo', size_by='avg_cost', title='t-SNE Provider Clustering'):
    import matplotlib.patches as mpatches

    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        data=tsne_df,
        x='TSNE1',
        y='TSNE2',
        hue=color_by,
        size=size_by,
        sizes=(8, 40),
        palette='tab10',
        alpha=0.7,
        legend=False   # Do NOT show legend on the main plot
    )
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.tight_layout()
    plt.savefig('tsne_provider_chronic_combo_interpret.png')
    plt.show()

    # --- Create a separate legend window ---
    unique_labels = tsne_df[color_by].unique()
    palette = sns.color_palette('tab10', len(unique_labels))
    handles = [
        mpatches.Patch(color=palette[i], label=str(lbl))
        for i, lbl in enumerate(unique_labels)
    ]
    legend_fig = plt.figure(figsize=(3, min(0.5 * len(unique_labels), 10)))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.legend(handles=handles, loc='center', frameon=False)
    legend_ax.axis('off')
    plt.title(f'Legend for {color_by} (color mapping)')
    plt.tight_layout()
    plt.savefig('tsne_chronic_combo_color_legend.png')
    plt.show()



def label_tsne_outliers(tsne_df, threshold_percentile=95, offset=3):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    threshold = np.percentile(tsne_df['avg_cost'], threshold_percentile)
    big = tsne_df[tsne_df['avg_cost'] >= threshold]
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        data=tsne_df,
        x='TSNE1',
        y='TSNE2',
        hue='ChronicCombo',
        size='avg_cost',
        sizes=(50, 400),
        palette='tab10',
        alpha=0.8,
        legend='brief'
    )
    # Overlay black boxes for outliers
    ax.scatter(
        big['TSNE1'], big['TSNE2'],
        s=130,                   # Box size
        c='black',
        marker='s',
        label='Priority Outlier'
    )
    plt.title('t-SNE Provider Clustering (Priority Outliers = Black Boxes)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.savefig('tsne_provider_chronic_combo_outliers.png')
    plt.show()




# --- Run t-SNE analysis and plot ---
tsne_df = tsne_provider_clustering(merged)
tsne_df.to_csv('tsne.csv')
visualize_tsne_clusters(tsne_df)
label_tsne_outliers(tsne_df)
