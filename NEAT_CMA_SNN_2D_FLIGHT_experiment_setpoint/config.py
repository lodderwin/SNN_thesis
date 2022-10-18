# Configurations for NEAT


INPUT_NEURONS = 12
OUTPUT_NEURONS = 4
POPULATION = 250
# True = population is const across all species and changing
# False = genomes may still move species, but each new species gets POPULATION starting genomes
DYNAMIC_POPULATION = True
STAGNATED_SPECIES_THRESHOLD = 500
STAGNATIONS_ALLOWED = 200
SPECIATION = True
CROSSOVER_CHANCE = 0.0 # 0.0 means no crossover, all cloning mutations, default 0.75
WEAK_SPECIES_THRESHOLD = 10  #orignal 5
ACTIVATION_THRESHOLD = 0.5
WEIGHT_MUTATION_RATE = 0.2
UNIFORM_WEIGHT_MUTATION_RATE = 0.9
ADD_GENE_MUTATION = 0.3 # Default 0.05
ADD_NODE_MUTATION = 0.25# Default 0.03
ENABLE_GENE_MUTATION_RATE = 0.1 # 0.0 means no reenabling of genes possible
INHERIT_DISABLED_GENE_RATE = 0.2
COMPATIBILITY_THRESHOLD = 100.0  #original 3
EXCESS_COMPATIBILITY_CONSTANT = 1.0
DISJOINT_COMPATIBILITY_CONSTANT = 1.0
WEIGHT_COMPATIBILITY_CONSTANT = 0.4



#Added later stage
V_DECAY_MUTATION = 0.2
THRESHOLD_MUTATION = 0.18

GENE_REMOVE_RATE = 1.0
