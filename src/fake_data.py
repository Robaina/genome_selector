import numpy as np
import pandas as pd


def generate_fake_data(
    n_genomes=100, n_ec_numbers=1000, min_ec=700, max_ec=900, abundance_scale=20
):
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
        {"id_genome": genome_ids, "relative_abundance": abundances}
    ).set_index("id_genome")

    return ec_df, abundance_df
