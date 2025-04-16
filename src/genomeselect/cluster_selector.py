from collections import defaultdict
from typing import Dict, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import time


def select_representative_genomes_clustering(
    trait_data: pd.DataFrame,
    abundance_data: pd.DataFrame,
    n_clusters: int = 20,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Select representative genomes using hierarchical clustering based on trait profiles.

    Args:
        trait_data (pd.DataFrame): DataFrame containing trait presence/absence data for all genomes.
        abundance_data (pd.DataFrame): DataFrame containing abundance data for all genomes.
        n_clusters (int): Number of clusters to create (default: 20).
        verbose (bool): Whether to print information during the process (default: True).

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping selected genome IDs to a dictionary containing:
            - 'abundance': The abundance value
            - 'trait_count': Number of traits covered by the genome
            - 'trait_percentage': Percentage of traits covered relative to total non-zero traits
    """
    # Preprocess: Ensure both datasets have the same genomes
    common_genomes = list(set(trait_data.index) & set(abundance_data.index))
    trait_data = trait_data.loc[common_genomes]
    abundance_data = abundance_data.loc[common_genomes]
    
    # Preprocess: Remove columns (traits) with zero sum
    non_zero_traits = trait_data.columns[trait_data.sum() > 0]
    trait_data = trait_data[non_zero_traits]
    
    # Preprocess: Remove rows (genomes) with zero sum
    non_zero_genomes = trait_data.index[trait_data.sum(axis=1) > 0]
    trait_data = trait_data.loc[non_zero_genomes]
    abundance_data = abundance_data.loc[non_zero_genomes]
    
    # Get the total number of traits for percentage calculation
    total_traits = len(non_zero_traits)
    
    # Check if there's only one genome
    if len(non_zero_genomes) == 1:
        genome = non_zero_genomes[0]
        if verbose:
            print(f"Warning: Only one genome found in the input dataset: {genome}")
        
        # Calculate traits for this single genome
        genome_traits = trait_data.loc[genome].sum()
        trait_count = int(genome_traits)
        trait_percentage = (trait_count / total_traits) * 100 if total_traits > 0 else 0.0
        
        return {
            genome: {
                'abundance': float(abundance_data.loc[genome, "abundance"]),
                'trait_count': trait_count,
                'trait_percentage': float(trait_percentage)
            }
        }

    if verbose:
        print(f"Number of genomes after preprocessing: {len(non_zero_genomes)}")
        print(f"Number of traits after preprocessing: {len(non_zero_traits)}")

    # Compute pairwise distances using Jaccard metric
    distances = pdist(trait_data.values, metric="jaccard")

    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method="average")

    # Cut the dendrogram to obtain the desired number of clusters
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    # Group genomes by cluster
    clusters = defaultdict(list)
    for genome, label in zip(trait_data.index, cluster_labels):
        clusters[label].append(genome)

    # Select representative genome from each cluster
    selected_genomes = {}
    for cluster in clusters.values():
        # Select the genome with the highest abundance in the cluster
        representative = max(cluster, key=lambda g: abundance_data.loc[g, "abundance"])
        
        # Calculate traits for this genome
        genome_traits = trait_data.loc[representative].sum()
        trait_count = int(genome_traits)
        trait_percentage = (trait_count / total_traits) * 100 if total_traits > 0 else 0.0
        
        selected_genomes[representative] = {
            'abundance': float(abundance_data.loc[representative, "abundance"]),
            'trait_count': trait_count,
            'trait_percentage': float(trait_percentage)
        }

    # Calculate total trait coverage across all selected genomes
    all_selected_traits = set()
    for genome in selected_genomes:
        genome_traits = trait_data.columns[trait_data.loc[genome] == 1]
        all_selected_traits.update(genome_traits)

    if verbose:
        print(f"Total traits covered across all selected genomes: {len(all_selected_traits)}")
        total_coverage_percentage = (len(all_selected_traits) / total_traits) * 100 if total_traits > 0 else 0.0
        print(f"Percentage of traits covered: {total_coverage_percentage:.2f}%")

    return selected_genomes


def plot_trait_coverage_venn(
    trait_data: pd.DataFrame,
    selected_genomes: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Create a Venn diagram showing the overlap of traits between
    all genomes and the selected representative genomes.

    Args:
        trait_data (pd.DataFrame): DataFrame containing trait presence/absence data for all genomes.
        selected_genomes (Dict[str, Dict[str, float]]): Dictionary mapping selected representative genome IDs 
                                                      to dictionaries containing abundance and trait information.
        output_path (Optional[str]): Path to save the figure. If None, the figure is not saved.
        figsize (Optional[Tuple[float, float]]): Figure size (width, height) in inches. If None, default size is used.

    Returns:
        None
    """
    all_traits = set(trait_data.columns[trait_data.sum() > 0])
    selected_traits = set(
        trait_data.loc[list(selected_genomes.keys())].columns[trait_data.loc[list(selected_genomes.keys())].sum() > 0]
    )

    plt.figure(figsize=figsize or (10, 6))
    venn2([all_traits, selected_traits], set_labels=("All Genomes", "Selected Genomes"))
    plt.title("Trait Coverage")

    if output_path:
        plt.savefig(output_path)

    plt.show()

# Helper functions must be defined at module level for multiprocessing to work
_TRAIT_DATA = None
_ABUNDANCE_DATA = None
_TOTAL_ABUNDANCE = None

def _compute_metrics_for_n(n):
    """Compute trait count and cumulative abundance for n clusters."""
    try:
        selected = select_representative_genomes_clustering(
            _TRAIT_DATA, _ABUNDANCE_DATA, n_clusters=n, verbose=False
        )
        
        if len(selected) < n:
            return n, None, None  # Signal that we've reached maximum clusters
        
        # Extract data from the updated dictionary structure
        genome_keys = list(selected.keys())
        
        # Vectorized computation of trait count
        trait_count = _TRAIT_DATA.loc[genome_keys].sum().astype(bool).sum()
        
        # Sum the abundances from the dictionary values
        cum_abundance = sum(genome_info['abundance'] for genome_info in selected.values()) / _TOTAL_ABUNDANCE * 100
        
        return n, trait_count, cum_abundance
    except Exception as e:
        print(f"Error processing n={n}: {e}")
        return n, None, None

def plot_trait_cumulative(
    trait_data: pd.DataFrame,
    abundance_data: pd.DataFrame,
    max_genomes: Optional[int] = None,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    n_jobs: int = 1,  # Default to single process to avoid pickling issues
) -> None:
    """
    Create a comprehensive plot showing the number of represented traits
    and the cumulative abundance versus the number of selected genomes.
    
    Args:
        trait_data (pd.DataFrame): DataFrame containing trait presence/absence data for all genomes.
        abundance_data (pd.DataFrame): DataFrame containing abundance data for all genomes.
        max_genomes (Optional[int]): Maximum number of genomes to consider. If None, all genomes are considered.
        output_path (Optional[str]): Path to save the figure. If None, the figure is not saved.
        figsize (Optional[Tuple[float, float]]): Figure size (width, height) in inches. If None, default size is used.
        n_jobs (int): Number of processes to use for parallel computation. Default is 1.
    
    Returns:
        None
    """
    # Step 1: Calculate the total number of genomes to process
    total_genomes = min(len(trait_data), len(abundance_data))
    actual_genomes = total_genomes if max_genomes is None else min(total_genomes, max_genomes)
    
    # Pre-compute total abundance once
    total_abundance = abundance_data["abundance"].sum()
    
    # Step 2: Sequential processing (but with caching for efficiency)
    # We'll use a cache to avoid redundant clustering calculations
    results = {}
    max_selected = 0
    
    # Use global variables to store data needed for computation if using multiprocessing
    global _TRAIT_DATA, _ABUNDANCE_DATA, _TOTAL_ABUNDANCE
    _TRAIT_DATA = trait_data
    _ABUNDANCE_DATA = abundance_data
    _TOTAL_ABUNDANCE = total_abundance
    
    # Use multiprocessing only if n_jobs > 1 and the environment supports it
    if n_jobs > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            # Process in batches to avoid memory issues with very large datasets
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Create futures for each n
                futures = [executor.submit(_compute_metrics_for_n, n) for n in range(1, actual_genomes + 1)]
                
                # Process results as they complete
                for future in as_completed(futures):
                    n, trait_count, cum_abundance = future.result()
                    
                    if trait_count is None:  # We've reached the maximum number of clusters
                        continue
                        
                    results[n] = (trait_count, cum_abundance)
                    max_selected = max(max_selected, n)
                    
        except Exception as e:
            print(f"Parallel processing failed: {e}. Falling back to sequential execution.")
            # Fall back to sequential processing
            n_jobs = 1
    
    # Sequential processing if n_jobs == 1 or parallel processing failed
    if n_jobs == 1:
        print("Using sequential processing...")
        # Cache for select_representative_genomes_clustering results
        cluster_cache = {}
        
        for n in range(1, actual_genomes + 1):
            start_time = time.time()
            
            # Get clusters, using cache if possible
            if n in cluster_cache:
                selected = cluster_cache[n]
            else:
                selected = select_representative_genomes_clustering(
                    trait_data, abundance_data, n_clusters=n, verbose=False
                )
                cluster_cache[n] = selected
            
            if len(selected) < n:
                break  # We've reached the maximum number of clusters
            
            # Calculate metrics for the new dictionary structure
            trait_count = trait_data.loc[list(selected.keys())].sum().astype(bool).sum()
            cum_abundance = sum(genome_info['abundance'] for genome_info in selected.values()) / total_abundance * 100
            
            results[n] = (trait_count, cum_abundance)
            max_selected = max(max_selected, n)
            
            elapsed = time.time() - start_time
            if n % 10 == 0:
                print(f"Processed n={n} in {elapsed:.2f} seconds")
    
    # Step 5: Sort results and prepare plotting data
    sorted_ns = sorted(results.keys())
    trait_counts = [results[n][0] for n in sorted_ns]
    cumulative_abundance = [results[n][1] for n in sorted_ns]
    
    # Step 6: Create the plot
    fig, ax1 = plt.subplots(figsize=figsize or (12, 8))
    
    # Plot trait counts
    color = "tab:blue"
    ax1.set_xlabel("Number of Selected Genomes")
    ax1.set_ylabel("Number of Represented Traits", color=color)
    ax1.plot(sorted_ns, trait_counts, color=color, marker="o")
    ax1.tick_params(axis="y", labelcolor=color)
    
    # Plot cumulative abundance
    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Cumulative Abundance of Selected Genomes (%)", color=color)
    ax2.plot(sorted_ns, cumulative_abundance, color=color, marker="^")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 100)  # Set y-axis limits from 0 to 100%
    
    plt.title("Comprehensive View of Genome Selection Process")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Set x-axis ticks and labels
    x_ticks = np.linspace(min(sorted_ns), max(sorted_ns), 
                         min(10, len(sorted_ns)), dtype=int)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(x) for x in x_ticks])
    
    # Adjust x-axis limit to match the actual number of genomes
    ax1.set_xlim(min(sorted_ns) - 0.5, max(sorted_ns) + 0.5)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        ["Traits", "Cumulative Abundance"],
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    
    plt.show()
    
    return sorted_ns, trait_counts, cumulative_abundance


# Alternative implementation using batched incremental approach
# This approach is useful if the clusters can be built incrementally
def plot_trait_cumulative_incremental(
    trait_data: pd.DataFrame,
    abundance_data: pd.DataFrame,
    max_genomes: Optional[int] = None,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Create a comprehensive plot using an incremental approach to clustering.
    This works if your clustering algorithm supports incremental updates.
    
    This implementation assumes you have or can create a modified version of 
    select_representative_genomes_clustering that can add genomes incrementally.
    """
    total_genomes = min(len(trait_data), len(abundance_data))
    actual_genomes = total_genomes if max_genomes is None else min(total_genomes, max_genomes)
    
    total_abundance = abundance_data["abundance"].sum()
    
    # Function to select the next best genome to add to the selection
    def select_next_genome(current_selection):
        """
        This is a placeholder for an incremental genome selection function.
        You would need to implement this based on your specific clustering algorithm.
        """
        # This is where you would implement logic to select the next genome
        # based on the current selection
        remaining_genomes = set(trait_data.index) - set(current_selection)
        if not remaining_genomes:
            return None
            
        # Example logic (this should be replaced with your actual algorithm)
        # Find the genome that adds the most new traits
        best_genome = None
        max_new_traits = -1
        
        current_traits = set()
        if current_selection:
            current_traits = set(trait_data.loc[current_selection].sum().where(lambda x: x > 0).dropna().index)
        
        for genome in remaining_genomes:
            genome_traits = set(trait_data.loc[[genome]].sum().where(lambda x: x > 0).dropna().index)
            new_traits = len(genome_traits - current_traits)
            
            if new_traits > max_new_traits:
                max_new_traits = new_traits
                best_genome = genome
                
        return best_genome
    
    # Incrementally build up the selection
    selected_genomes = {}  # Changed to dict to match the new return type
    trait_counts = []
    cumulative_abundance = []
    
    # Cache for trait data to avoid redundant computations
    trait_presence_cache = {}
    
    for n in range(1, actual_genomes + 1):
        # Get the next genome to add
        next_genome = select_next_genome(list(selected_genomes.keys()))
        
        if next_genome is None:
            break
            
        # Store genome with its metrics in a nested dictionary
        genome_traits = trait_data.loc[next_genome].sum()
        trait_count = int(genome_traits)
        trait_percentage = (trait_count / len(trait_data.columns)) * 100 if len(trait_data.columns) > 0 else 0.0
        
        selected_genomes[next_genome] = {
            'abundance': float(abundance_data.loc[next_genome, "abundance"]),
            'trait_count': trait_count,
            'trait_percentage': float(trait_percentage)
        }
        
        # Efficiently compute trait count using cache
        if n == 1:
            # First genome - compute directly
            trait_presence = trait_data.loc[[next_genome]].sum().astype(bool)
            trait_presence_cache[next_genome] = set(trait_presence[trait_presence].index)
            trait_count = len(trait_presence_cache[next_genome])
        else:
            # Add new traits from this genome to the set
            new_traits = trait_presence_cache[next_genome] if next_genome in trait_presence_cache else set(
                trait_data.loc[[next_genome]].sum().astype(bool)[lambda x: x].index
            )
            
            # Update the total set of traits
            all_traits = set().union(*(trait_presence_cache[g] for g in selected_genomes.keys() if g in trait_presence_cache))
            trait_count = len(all_traits)
        
        trait_counts.append(trait_count)
        
        # Calculate cumulative abundance from the nested dictionary
        cum_abundance = sum(genome_info['abundance'] for genome_info in selected_genomes.values()) / total_abundance * 100
        cumulative_abundance.append(cum_abundance)
    
    # Plot using the collected data
    # (Plot code is the same as in the previous function)
    fig, ax1 = plt.subplots(figsize=figsize or (12, 8))
    
    # Plot trait counts
    color = "tab:blue"
    ax1.set_xlabel("Number of Selected Genomes")
    ax1.set_ylabel("Number of Represented Traits", color=color)
    ax1.plot(range(1, len(trait_counts) + 1), trait_counts, color=color, marker="o")
    ax1.tick_params(axis="y", labelcolor=color)
    
    # Plot cumulative abundance
    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Cumulative Abundance of Selected Genomes (%)", color=color)
    ax2.plot(range(1, len(cumulative_abundance) + 1), cumulative_abundance, color=color, marker="^")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 100)  # Set y-axis limits from 0 to 100%
    
    plt.title("Comprehensive View of Genome Selection Process")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Set x-axis ticks and labels
    max_selected = len(trait_counts)
    x_ticks = range(1, max_selected + 1, max(1, max_selected // 10))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(x) for x in x_ticks])
    
    # Adjust x-axis limit to match the actual number of genomes
    ax1.set_xlim(0.5, max_selected + 0.5)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        ["Traits", "Cumulative Abundance"],
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    
    plt.show()
    
    # Return the dict of selected genomes with their abundances, and the analysis data
    return selected_genomes, trait_counts, cumulative_abundance