from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def select_representative_genomes_clustering(
    ec_data: pd.DataFrame,
    abundance_data: pd.DataFrame,
    n_clusters: int = 20,
    verbose: bool = True,
) -> List[str]:
    """
    Select representative genomes using hierarchical clustering based on EC number profiles.

    Args:
        ec_data (pd.DataFrame): DataFrame containing EC number presence/absence data for all genomes.
        abundance_data (pd.DataFrame): DataFrame containing abundance data for all genomes.
        n_clusters (int): Number of clusters to create (default: 20).
        verbose (bool): Whether to print information during the process (default: True).

    Returns:
        List[str]: List of selected representative genome IDs.
    """
    # Ensure both datasets have the same genomes
    common_genomes = list(set(ec_data.index) & set(abundance_data.index))
    ec_data = ec_data.loc[common_genomes]
    abundance_data = abundance_data.loc[common_genomes]

    if verbose:
        print(f"Number of genomes: {len(common_genomes)}")
        print(f"Number of EC numbers: {ec_data.shape[1]}")

    # Compute pairwise distances using Jaccard metric
    distances = pdist(ec_data.values, metric="jaccard")

    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method="average")

    # Cut the dendrogram to obtain the desired number of clusters
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    # Group genomes by cluster
    clusters = defaultdict(list)
    for genome, label in zip(ec_data.index, cluster_labels):
        clusters[label].append(genome)

    # Select representative genome from each cluster
    selected_genomes = []
    for cluster in clusters.values():
        # Select the genome with the highest abundance in the cluster
        representative = max(cluster, key=lambda g: abundance_data.loc[g, "abundance"])
        selected_genomes.append(representative)

    # Calculate EC number coverage
    covered_ec = set()
    for genome in selected_genomes:
        covered_ec.update(ec_data.columns[ec_data.loc[genome] == 1])

    if verbose:
        print(f"Total EC numbers covered: {len(covered_ec)}")

    return selected_genomes


def plot_ec_coverage_venn(
    ec_data: pd.DataFrame,
    selected_genomes: List[str],
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Create a Venn diagram showing the overlap of EC numbers between
    all genomes and the selected representative genomes.

    Args:
        ec_data (pd.DataFrame): DataFrame containing EC number presence/absence data for all genomes.
        selected_genomes (List[str]): List of genome IDs selected as representatives.
        output_path (Optional[str]): Path to save the figure. If None, the figure is not saved.
        figsize (Optional[Tuple[float, float]]): Figure size (width, height) in inches. If None, default size is used.

    Returns:
        None
    """
    all_ec = set(ec_data.columns[ec_data.sum() > 0])
    selected_ec = set(
        ec_data.loc[selected_genomes].columns[ec_data.loc[selected_genomes].sum() > 0]
    )

    plt.figure(figsize=figsize or (10, 6))
    venn2([all_ec, selected_ec], set_labels=("All Genomes", "Selected Genomes"))
    plt.title("EC Number Coverage")

    if output_path:
        plt.savefig(output_path)

    plt.show()


def plot_ec_cumulative(
    ec_data: pd.DataFrame,
    abundance_data: pd.DataFrame,
    max_genomes: Optional[int] = None,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Create a comprehensive plot showing the number of represented EC numbers
    and the cumulative abundance versus the number of selected genomes.

    Args:
        ec_data (pd.DataFrame): DataFrame containing EC number presence/absence data for all genomes.
        abundance_data (pd.DataFrame): DataFrame containing abundance data for all genomes.
        max_genomes (Optional[int]): Maximum number of genomes to consider. If None, all genomes are considered.
        output_path (Optional[str]): Path to save the figure. If None, the figure is not saved.
        figsize (Optional[Tuple[float, float]]): Figure size (width, height) in inches. If None, default size is used.

    Returns:
        None
    """
    total_genomes = min(len(ec_data), len(abundance_data))
    actual_genomes = (
        total_genomes if max_genomes is None else min(total_genomes, max_genomes)
    )

    ec_counts, cumulative_abundance = [], []

    total_abundance = abundance_data["abundance"].sum()

    # Determine the maximum number of genomes that can be selected
    max_selected = 0
    for n in range(1, actual_genomes + 1):
        selected = select_representative_genomes_clustering(
            ec_data, abundance_data, n_clusters=n, verbose=False
        )
        if len(selected) < n:
            max_selected = n - 1
            break
        max_selected = n

        # Calculate EC count
        ec_count = ec_data.loc[selected].sum().astype(bool).sum()
        ec_counts.append(ec_count)

        # Calculate cumulative abundance of selected genomes
        cum_abundance = (
            abundance_data.loc[selected, "abundance"].sum() / total_abundance * 100
        )
        cumulative_abundance.append(cum_abundance)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=figsize or (12, 8))

    # Plot EC counts
    color = "tab:blue"
    ax1.set_xlabel("Number of Selected Genomes")
    ax1.set_ylabel("Number of Represented EC Numbers", color=color)
    ax1.plot(range(1, max_selected + 1), ec_counts, color=color, marker="o")
    ax1.tick_params(axis="y", labelcolor=color)

    # Plot cumulative abundance
    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Cumulative Abundance of Selected Genomes (%)", color=color)
    ax2.plot(range(1, max_selected + 1), cumulative_abundance, color=color, marker="^")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 100)  # Set y-axis limits from 0 to 100%

    plt.title("Comprehensive View of Genome Selection Process")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set x-axis ticks and labels
    x_ticks = range(
        1, max_selected + 1, max(1, max_selected // 10)
    )  # Show about 10 ticks
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(x) for x in x_ticks])

    # Adjust x-axis limit to match the actual number of genomes
    ax1.set_xlim(0.5, max_selected + 0.5)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        ["EC Numbers", "Cumulative Abundance"],
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()
