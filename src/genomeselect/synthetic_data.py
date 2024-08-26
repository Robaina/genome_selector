import numpy as np
import pandas as pd
from typing import Tuple


def generate_synthetic_data(
    n_genomes: int = 100,
    n_ec_numbers: int = 1000,
    min_ec: int = 700,
    max_ec: int = 900,
    abundance_scale: float = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate fake data for testing cGEM (community Genome-scale Metabolic model) selection algorithms.

    This function creates two DataFrames:
    1. EC number presence/absence data for each genome
    2. Abundance data for each genome

    Args:
        n_genomes (int): Number of genomes to generate. Default is 100.
        n_ec_numbers (int): Number of EC numbers to consider. Default is 1000.
        min_ec (int): Minimum number of EC numbers present in each genome. Default is 700.
        max_ec (int): Maximum number of EC numbers present in each genome. Default is 900.
        abundance_scale (float): Scale parameter for the exponential distribution used to generate abundances. Default is 20.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - EC number presence/absence data (index: genome_id, columns: EC numbers)
            - Abundance data (index: genome_id, column: abundance)
    """
    # Generate EC number presence/absence data
    ec_data = np.zeros((n_genomes, n_ec_numbers), dtype=int)

    for i in range(n_genomes):
        n_present = np.random.randint(min_ec, max_ec + 1)
        ec_present = np.random.choice(n_ec_numbers, size=n_present, replace=False)
        ec_data[i, ec_present] = 1

    # Create DataFrame for EC data
    genome_ids = [f"genome_{i+1}" for i in range(n_genomes)]
    ec_numbers = [f"EC_{i+1}" for i in range(n_ec_numbers)]
    ec_df = pd.DataFrame(ec_data, index=genome_ids, columns=ec_numbers)
    ec_df.index.name = "id_genome"

    # Generate abundance data (exponential distribution with long tail)
    abundances = np.random.exponential(scale=abundance_scale, size=n_genomes)

    # Normalize abundances to sum to 100
    abundances = (abundances / abundances.sum()) * 100

    # Create DataFrame for abundance data
    abundance_df = pd.DataFrame(
        {"id_genome": genome_ids, "abundance": abundances}
    ).set_index("id_genome")

    return ec_df, abundance_df
