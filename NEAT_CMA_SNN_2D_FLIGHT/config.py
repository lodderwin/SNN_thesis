# Configurations for NEAT


INPUT_NEURONS = 10
OUTPUT_NEURONS = 4
POPULATION = 25
# True = population is const across all species and changing
# False = genomes may still move species, but each new species gets POPULATION starting genomes
DYNAMIC_POPULATION = True
STAGNATED_SPECIES_THRESHOLD = 20
STAGNATIONS_ALLOWED = 2
SPECIATION = True
CROSSOVER_CHANCE = 0.1 # 0.0 means no crossover, all cloning mutations, default 0.75
WEAK_SPECIES_THRESHOLD = 5  #orignal 5
ACTIVATION_THRESHOLD = 0.5
WEIGHT_MUTATION_RATE = 0.8
UNIFORM_WEIGHT_MUTATION_RATE = 0.9
ADD_GENE_MUTATION = 0.1 # Default 0.05
ADD_NODE_MUTATION = 0.07 # Default 0.03
ENABLE_GENE_MUTATION_RATE = 0.1 # 0.0 means no reenabling of genes possible
INHERIT_DISABLED_GENE_RATE = 0.2
COMPATIBILITY_THRESHOLD = 3.0
EXCESS_COMPATIBILITY_CONSTANT = 1.0
DISJOINT_COMPATIBILITY_CONSTANT = 1.0
WEIGHT_COMPATIBILITY_CONSTANT = 0.4



#Added later stage
V_DECAY_MUTATION = 0.5
THRESHOLD_MUTATION = 0.4
