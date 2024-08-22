from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import defaultdict


def select_representative_genomes_clustering(ec_data, abundance_data, n_clusters=20):
    # Ensure both datasets have the same genomes
    common_genomes = list(set(ec_data.index) & set(abundance_data.index))
    ec_data = ec_data.loc[common_genomes]
    abundance_data = abundance_data.loc[common_genomes]

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

    print(f"Total EC numbers covered: {len(covered_ec)}")
    return selected_genomes
