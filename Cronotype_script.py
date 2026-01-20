import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="WGCNA-like analysis with clade and module outputs.")
    parser.add_argument("-i", "--input", required=True, help="Input read count matrix with 'Clade' column")
    parser.add_argument("-o", "--output_prefix", required=True, help="Prefix for output files")
    parser.add_argument("-t", "--threshold", type=float, default=0.8, help="Correlation threshold for network edges")
    parser.add_argument("-d", "--distance_cutoff", type=float, default=0.7, help="Dendrogram distance cutoff for modules")
    parser.add_argument("--grey_cutoff", type=float, default=0.3, help="Average intra-module correlation cutoff for Grey module")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load input file
    df = pd.read_csv(args.input, index_col=0)
    assert 'Clade' in df.columns, "The input file must have a 'Clade' column"
    
    clades = df['Clade']
    df = df.drop(columns=['Clade'])

    # Transpose if genes are in rows
    if df.shape[0] > df.shape[1]:
        df = df.T

    gene_names = df.columns

    # Compute correlation and distance matrices
    corr = df.corr(method='pearson')
    np.fill_diagonal(corr.values, 0)
    distance = 1 - corr
    np.fill_diagonal(distance.values, 0)
    dist_array = squareform(distance.values)
    linkage_matrix = linkage(dist_array, method='average')

    # Dendrogram with clade labels
    clade_map = clades.to_dict()
    unique_clades = clades.unique()
    clade_palette = sns.color_palette("tab10", len(unique_clades))
    clade_to_color = dict(zip(unique_clades, clade_palette))

    plt.figure(figsize=(12, 6))
    dendro = dendrogram(linkage_matrix, labels=gene_names, leaf_rotation=90)
    leaf_colors = [clade_to_color.get(clade_map.get(label, ""), "black") for label in dendro['ivl']]
    for xtick, color in zip(plt.gca().get_xticklabels(), leaf_colors):
        xtick.set_color(color)
    handles = [mpatches.Patch(color=clade_to_color[c], label=c) for c in unique_clades]
    plt.legend(handles=handles, title="Clades", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Dendrogram with Clade Labels")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_clade_dendrogram.png")
    plt.close()

    # Detect modules from dendrogram
    module_labels = fcluster(linkage_matrix, t=args.distance_cutoff, criterion='distance')
    modules = pd.Series(module_labels, index=gene_names, name="Module")

    # Reassign genes to Grey if intra-module correlation is low
    grey_genes = []
    for gene in gene_names:
        module_id = modules[gene]
        peers = modules[modules == module_id].index
        if len(peers) <= 1:
            grey_genes.append(gene)
            continue
        avg_corr = corr.loc[gene, peers].mean()
        if avg_corr < args.grey_cutoff:
            grey_genes.append(gene)
    modules.loc[grey_genes] = 0  # Assign to Grey module (Module 0)

    # Recalculate unique modules
    unique_modules = sorted(modules.unique())
    module_palette = sns.color_palette("hls", len([m for m in unique_modules if m != 0]))
    module_color_map = {m: c for m, c in zip([m for m in unique_modules if m != 0], module_palette)}
    module_color_map[0] = "#B0B0B0"  # Grey module color
    module_colors = modules.map(module_color_map)

    # Dendrogram with module colors
    plt.figure(figsize=(12, 6))
    dendro = dendrogram(linkage_matrix, labels=gene_names, leaf_rotation=90)
    mod_leaf_colors = [module_color_map[modules[label]] for label in dendro['ivl']]
    for xtick, color in zip(plt.gca().get_xticklabels(), mod_leaf_colors):
        xtick.set_color(color)
    handles = [mpatches.Patch(color=module_color_map[m], label=f"Module {m}" if m != 0 else "Grey") for m in unique_modules]
    plt.legend(handles=handles, title="Modules", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Dendrogram with Module Colors")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_module_dendrogram.png")
    plt.close()

    # Calculate module eigengenes
    eigengenes = {}
    for module_id in unique_modules:
        if module_id == 0:
            continue  # Skip Grey
        module_genes = modules[modules == module_id].index
        module_data = df[module_genes]

        if module_data.shape[1] > 1:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(module_data)
            pca = PCA(n_components=1)
            eigengenes[module_id] = pca.fit_transform(scaled_data).flatten()
        else:
            eigengenes[module_id] = module_data.iloc[:, 0].values

    eigengene_df = pd.DataFrame(eigengenes, index=df.index)
    eigengene_df.columns = [f"ME{m}" for m in eigengene_df.columns]
    eigengene_df.to_csv(f"{args.output_prefix}_module_eigengenes.csv")

    # Save module assignments
    modules.to_csv(f"{args.output_prefix}_module_assignments.csv", header=True)

    # Build correlation network
    G = nx.Graph()
    for gene in gene_names:
        G.add_node(gene, clade=clade_map.get(gene, "Unknown"), module=int(modules.get(gene, 0)))

    for i in range(len(gene_names)):
        for j in range(i + 1, len(gene_names)):
            g1, g2 = gene_names[i], gene_names[j]
            corr_val = corr.loc[g1, g2]
            if corr_val >= args.threshold:
                G.add_edge(g1, g2, weight=corr_val)

    # Export network
    edges_df = nx.to_pandas_edgelist(G)
    nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    edges_df.to_csv(f"{args.output_prefix}_network_edges.csv", index=False)
    nodes_df.to_csv(f"{args.output_prefix}_network_nodes.csv")

    print(f"Finished. Output written with prefix '{args.output_prefix}'.")


if __name__ == "__main__":
    main()
