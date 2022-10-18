# from black import NothingChanged
from matplotlib.pyplot import fill
import config as config

import torch
from gym import Env

from innovation import Innovation
from network import Network
from gene import Gene
from species import Species, find_all_routes, place_weights, clean_array
from network_viz import draw_net
import networkx as nx
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import dill
# objective function in neat Env
#genome = network


from visualize import DrawNN


class NEAT(object):

    def __init__(self):
        self.solved = False
        self.solution_genome = None

        self.population_N = config.POPULATION

        self.initial_genome_topology = (config.INPUT_NEURONS, config.OUTPUT_NEURONS)

        self.species_N = 0
        self.species = {} 

        self.population_fitness_avg = 0

        self.innovation = Innovation()

        initial_genome = Network(self.initial_genome_topology, self.innovation)
        # self.create_new_species(initial_genome, self.population_N)
        self.create_new_species_first(initial_genome, self.population_N)


        self.track_learning = []
        self.track_learning_cma= []

        self.best_species = None
        self.range_coordinates_x = [(-1,1), (1, 4), (1, 4), (-1, -4), (-1, -4)]
        self.range_coordinates_y = [(-1,1), (1, 4), (-1, -4), (-1, -4), (1, 4)]
        self.range_coordinates_z = [(2., 3.), (1., 3.5), (1., 3.5), (1., 3.5), (1., 3.5)]

        # xy hovering  ++ +- -- -+ a
        # z is randomized between 1 and 5.

        self.ref_x_pos = [np.random.uniform(self.range_coordinates_x[0][0], self.range_coordinates_x[0][1]), np.random.uniform(self.range_coordinates_x[1][0], self.range_coordinates_x[1][1]), np.random.uniform(self.range_coordinates_x[2][0], self.range_coordinates_x[2][1]), np.random.uniform(self.range_coordinates_x[3][0], self.range_coordinates_x[3][1]), np.random.uniform(self.range_coordinates_x[4][0], self.range_coordinates_x[4][1])]
        self.ref_y_pos = [np.random.uniform(self.range_coordinates_y[0][0], self.range_coordinates_y[0][1]), np.random.uniform(self.range_coordinates_y[1][0], self.range_coordinates_y[1][1]), np.random.uniform(self.range_coordinates_y[2][0], self.range_coordinates_y[2][1]), np.random.uniform(self.range_coordinates_y[3][0], self.range_coordinates_y[3][1]), np.random.uniform(self.range_coordinates_y[4][0], self.range_coordinates_y[4][1])]
        self.ref_z_pos = [np.random.uniform(self.range_coordinates_z[0][0], self.range_coordinates_z[0][1]), np.random.uniform(self.range_coordinates_z[1][0], self.range_coordinates_z[1][1]), np.random.uniform(self.range_coordinates_z[2][0], self.range_coordinates_z[2][1]), np.random.uniform(self.range_coordinates_z[3][0], self.range_coordinates_z[3][1]), np.random.uniform(self.range_coordinates_z[4][0], self.range_coordinates_z[4][1])]

    def start_evolutionary_process(self, iterations=1):
        i = 0
        # while not self.solved:
        while i<iterations:
            self.ref_x_pos = [np.random.uniform(self.range_coordinates_x[0][0], self.range_coordinates_x[0][1]), np.random.uniform(self.range_coordinates_x[1][0], self.range_coordinates_x[1][1]), np.random.uniform(self.range_coordinates_x[2][0], self.range_coordinates_x[2][1]), np.random.uniform(self.range_coordinates_x[3][0], self.range_coordinates_x[3][1]), np.random.uniform(self.range_coordinates_x[4][0], self.range_coordinates_x[4][1])]
            self.ref_y_pos = [np.random.uniform(self.range_coordinates_y[0][0], self.range_coordinates_y[0][1]), np.random.uniform(self.range_coordinates_y[1][0], self.range_coordinates_y[1][1]), np.random.uniform(self.range_coordinates_y[2][0], self.range_coordinates_y[2][1]), np.random.uniform(self.range_coordinates_y[3][0], self.range_coordinates_y[3][1]), np.random.uniform(self.range_coordinates_y[4][0], self.range_coordinates_y[4][1])]
            self.ref_z_pos = [np.random.uniform(self.range_coordinates_z[0][0], self.range_coordinates_z[0][1]), np.random.uniform(self.range_coordinates_z[1][0], self.range_coordinates_z[1][1]), np.random.uniform(self.range_coordinates_z[2][0], self.range_coordinates_z[2][1]), np.random.uniform(self.range_coordinates_z[3][0], self.range_coordinates_z[3][1]), np.random.uniform(self.range_coordinates_z[4][0], self.range_coordinates_z[4][1])]

            avg_fitness_scores = {}
            for s_id, s in self.species.items():
                avg_fitness = s.run_generation(i, self.ref_x_pos, self.ref_y_pos, self.ref_z_pos)
                if avg_fitness != None:
                    avg_fitness_scores[s_id] = avg_fitness
                    print('aaa', avg_fitness)

            if (len(avg_fitness_scores) == 0):
                print("\n\nAll species have gone extinct!\n\n")
                exit()

            
            if config.DYNAMIC_POPULATION:
                self.assign_species_populations_for_next_generation(avg_fitness_scores)

            self.best_species = max(avg_fitness_scores, key=avg_fitness_scores.get)
            self.best_genome = self.species[self.best_species].genomes[self.species[self.best_species].best_genome]
            self.track_learning.append(self.best_genome.fitness)

            # Evolve (create the next generation) for each species
            for s_id, s in self.species.items():
                s.evolve()   
           
            # Create new species from evolved current species

            # Turned off for now SPECIATION=FALSE
            if config.SPECIATION:
                self.perform_speciation()
            
            # break
            i+=1

        # Need to potentially set this somewhere...
            print('bbb', i)
        
        # maximum = max(mydict, key=mydict.get)  # Just use 'min' instead of 'max' for minimum.
        # print(maximum, mydict[maximum])

        
        # find best genome
        # best_performace = avg_fitness_scores[best_species]

        #   [x.fitness for x in self.species[best_species].genomes.values()]

        return self.solution_genome
    def first_round_evolutionary_process_neat(self):
        div_training = self.div_training
        wx_training = self.wx_training
        for s_id, s in self.species.items():
            s.first_round_evolutionary_process_species(div_training, wx_training)

    def assign_species_populations_for_next_generation(self, avg_fitness_scores):
        if len(avg_fitness_scores) == 1:
            return

        sorted_species_ids = sorted(avg_fitness_scores, key=avg_fitness_scores.get)

        # If any species were culled... reassign population to best species.
        active_pop = self.get_active_population()
        if active_pop < self.population_N:
            print("Active population:", active_pop)
            # print(sorted_species_ids)
            self.species[sorted_species_ids[0]].increment_population(self.population_N-active_pop)

        # Handle all other population changes.  # DIT KAN FOUT ZIJN! moet verhouding van species fitness tov totale gem fitness stijgen of dalen
        pop_change = int(math.floor(len(avg_fitness_scores)/2.0))
        start = 0
        end = len(sorted_species_ids) - 1
        while (start < end): 
            self.species[sorted_species_ids[start]].decrement_population(pop_change)
            self.species[sorted_species_ids[end]].increment_population(pop_change)
            start += 1
            end -= 1
            pop_change -= 1
            print('stuck here?')

    
    def perform_speciation(self):
        # here also runtime errror:
        # for s_id, s in self.species.items():
        for s_id, s in list(self.species.items()):
            # Only want to speciate and find evolve from active species
            if s.active:
                # changed because of rutimeerror: RuntimeError: dictionary changed size during iteration
                # for genome_index, genome in s.genomes.items():
                #     if not genome.is_compatible(s.species_genome_representative):
                #         self.assign_genome(genome, s_id)
                #         s.del ete_genome(genome_index)

                for genome_index, genome in list(s.genomes.items()):
                    if not genome.is_compatible(s.species_genome_representative):
                        self.assign_genome(genome, s_id)
                        s.delete_genome(genome_index)

                #hier gaat wat mis
    def assign_genome(self, genome, origin_species_id):
        # RuntimeError: dictionary changed size during iteration
        # for s_id, s in self.species.items():
        #     if genome.is_compatible(s.species_genome_representative):
        #         # If we add to dead species, it didn't deserve to live anyway
        #         s.add_genome(genome)  ~
        #         return
        
        for s_id, s in list(self.species.items()):
            if genome.is_compatible(s.species_genome_representative):
                # If we add to dead species, it didn't deserve to live anyway
                s.add_genome(genome)
                return

        # Not my favorite way of deciding on new populations...
        if config.DYNAMIC_POPULATION:
            new_species_pop = int(math.floor(self.species[origin_species_id].species_population/2.0))
            print('new species pop:', new_species_pop)
            origin_species_pop = int(math.ceil(self.species[origin_species_id].species_population/2.0))
            self.species[origin_species_id].set_population(origin_species_pop)
        else:
            new_species_pop = self.population_N

        self.create_new_species(genome, new_species_pop)


    def create_new_species(self, initial_species_genome, population):
        self.species[self.species_N] = Species(self.species_N, population, initial_species_genome)
        self.species_N += 1

    def create_new_species_first(self, initial_species_genome, population):
        # initial_genome = Network(self.initial_genome_topology, self.innovation)
        genomes = {i:initial_species_genome.clone() for i in range(population)}
        for i in range(1,population):
            genomes[i] = Network(self.initial_genome_topology, self.innovation)

        self.species[self.species_N] = Species(self.species_N, population, initial_species_genome, genomes)
        self.species_N += 1


    def get_active_population(self):
        active_population = 0
        for species in self.species.values():
            if species.active:
                active_population += species.species_population

        return active_population
import time

# either of the two
start_time = time.time() 
a = NEAT()
# a.first_round_evolutionary_process_neat()
a.start_evolutionary_process(iterations=1)
with open('testing_decreasedCMAES_3D_new_control_sys_faster.pkl', 'wb') as outp:
    dill.dump(a, outp)

print("--- %s seconds ---" % (time.time() - start_time))


with open('testing_decreasedCMAES_3D_new_control_sys_faster.pkl', 'rb') as f:
    a = dill.load(f)
for i in range(100):
    start_time = time.time()
    a.start_evolutionary_process(iterations=1)
    with open('testing_decreasedCMAES_3D_new_control_sys_faster.pkl', 'wb') as outp:
        dill.dump(a, outp)
    print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time() 

# a = NEAT()
# a.species[0].create_random_network(0, 50, 8) 
# network_viz = draw_net(a.species[0].genomes[0])

# network_viz.view()
# print("--- %s seconds ---" % (time.time() - start_time))

# draw_nn = DrawNN(model)
# draw_nn.draw()



# if neuron matrix more than blabla rows, don't add neuron


#end with last run with 100 div's, most versatile one

# matrix = find_all_routes(a.species[5].genomes[2])
# matrix = clean_array(matrix)
# model = place_weights(matrix,a.species[5].genomes[2] )



#er gaat iets fout met de volgorde van neuron en genes
#%%
# with open('company_data.pkl', 'rb') as f:
#     company1 = dill.load(f)
# with open('company_data.pkl', 'rb') as inp:
#     company1 = dill.load(inp)
#     print(company1.species)  # -> banana



# the_one = a.species[3].genomes[2]
# b = clean_array(find_all_routes(the_one))
# place_weights(b, the_one)





#     return model
#%%





    # place the neurons with only one possible position and redo cycle until only neuron with 2 or more possible positions exist
    
    # with np.where, see if distance between neurons (thus genes) are widely apart and place weights 1


        

        
# check check check

# find good network to check algorithm




# draw_net(a.species[0].genomes[3])

# find_all_routes(a.species[3].genomes[2])
# nx.draw_networkx(a.species[3].genomes[5].networkx_network)
# p = nx.shortest_path(a.species[1].genomes[5].networkx_network) 
# nx.dag_longest_path(a.species[1].genomes[5].networkx_network, weight='weight')
# take into account disabled genes
# delete all disabled genes for route calculations


#not in correct order oi


# find_all_routes(a.species[0].genomes[0])

# class Graph:
 
#     def __init__(self, V):
#         self.V = V
#         self.adj = [[] for i in range(V)]
 
#     def addEdge(self, u, v):
 
#         # Add v to u’s list.
#         self.adj[u].append(v)
 
#     # Returns count of paths from 's' to 'd'
#     def countPaths(self, s, d):
 
#         # Mark all the vertices
#         # as not visited
#         visited = [False] * self.V
 
#         # Call the recursive helper
#         # function to print all paths
#         pathCount = [0]
#         self.countPathsUtil(s, d, visited, pathCount)
#         return pathCount[0]
 
#     # A recursive function to print all paths
#     # from 'u' to 'd'. visited[] keeps track
#     # of vertices in current path. path[]
#     # stores actual vertices and path_index
#     # is current index in path[]
#     def countPathsUtil(self, u, d,
#                        visited, pathCount):
#         visited[u] = True
 
#         # If current vertex is same as
#         # destination, then increment count
#         if (u == d):
#             pathCount[0] += 1
 
#         # If current vertex is not destination
#         else:
 
#             # Recur for all the vertices
#             # adjacent to current vertex
#             i = 0
#             while i < len(self.adj[u]):
#                 if (not visited[self.adj[u][i]]):
#                     self.countPathsUtil(self.adj[u][i], d,
#                                         visited, pathCount)
#                 i += 1
 
#         visited[u] = False
 
 
# # Driver Code
# g = Graph(4)
# g.addEdge(0, 1)
# g.addEdge(0, 2)
# g.addEdge(0, 3)
# g.addEdge(2, 0)
# g.addEdge(2, 1)
# g.addEdge(1, 3)

# s = 2
# d = 3
# print(g.countPaths(s, d))

# a.species[0].genomes[0].neurons[0].id


# for visualization
# list of input neurons: [x.id for x in a.species[0].genomes[0].input_neurons]
# list of output neurons [x.id for x in a.species[0].genomes[0].output_neurons]




# %%


# neuron_matrix = find_all_routes(genome)
# neuron_matrix = clean_array(neuron_matrix)

# model = place_weights(neuron_matrix, genome)
# # environment = QuadHover()
# # objective_genome = objective(environment)
            
# reward = 0
# for i in range(len(div_training)):
#     reward += objective_genome.objective_function_NEAT(model, div_training[i], wx_training[i])
# reward = reward/float(len(div_training))