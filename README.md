## Select a subset of representative genomes from traits matrix and abundances


This package selects a genome subset of size _k_ from traits matrix (genomes vs presence/absence of enzymes) and relative abundances, such that functional diversity (enzyme set) is maximized and the selected genomes are representative of the entire community.

### Use case

This package was developed as a means to select representative genomes to build commnunity genome-scale metabolic models (cGEMs), representative of microbial communities found in metagenomic samples. cGEMs are composed of biochemical reactions at their core, hence the need to optimize for enzyme (i.e. reaction) diversity.