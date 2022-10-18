import torch 
import torch.nn as nn
from copy import deepcopy
from neuron import Neuron
from gene import Gene
import numpy as np
import config as config
from collections import Counter
import copy
import networkx as nx
# from collections import namedtuple
# import numpy as np
# from typing import Optional, NamedTuple, Tuple, Any, Sequence


class Network(object):
    def __init__(self, layer_lst, innovation):
        # super(Network, self).__init__()


        # monitor variables
        self.num_input_neurons = layer_lst[0]
        self.num_output_neurons = layer_lst[1]

        # dangerous!!!!!
        self.current_neuron_id = 1

        self.innovation = innovation

        self.genes = {}
        self.neurons = {}
        self.hidden_neurons = []

        self.edges = []

        # location properties input neurons
        locations_input_neurons = np.linspace(0., 1., int(self.num_input_neurons))
        i = 0
        self.input_neurons = []
        while i < self.num_input_neurons:
            new_neuron_id = self.get_next_neuron_id()
            self.neurons[new_neuron_id] = Neuron(new_neuron_id, x_position = 0., y_position = locations_input_neurons[i], n_type="Input")
            # self.neurons[new_neuron_id].x_location = 0.
            # self.neurons[new_neuron_id].y_location = locations_input_neurons[i]
            self.input_neurons.append(self.neurons[new_neuron_id])
            
            i += 1

        locations_output_neurons = np.linspace(0., 1., int(self.num_output_neurons))
        if int(self.num_output_neurons)==1:
            locations_output_neurons = [0.5]

        i = 0
        self.output_neurons = []
        while i < self.num_output_neurons:
            new_neuron_id = self.get_next_neuron_id()
            self.neurons[new_neuron_id] = Neuron(new_neuron_id, x_position = 1., y_position = locations_output_neurons[i], n_type="Output")
            # self.neurons[new_neuron_id].x_location = 1.
            # self.neurons[new_neuron_id].y_location = locations_input_neurons[i]
            self.output_neurons.append(self.neurons[new_neuron_id])
            i += 1


        # give new neurons coordinates

        # Create Genes of first network evah
        for input_neuron in self.input_neurons:
            for output_neuron in self.output_neurons:
                innov_num = self.innovation.get_new_innovation_number()
                self.genes[innov_num] = Gene(innov_num, input_neuron, output_neuron)
                # print(input_neuron,output_neuron)
                self.edges.append((input_neuron.id, output_neuron.id, 1.))

        # innov_num = self.innovation.get_new_innovation_number()
        # selected_input_node = np.random.choice(self.input_neurons)
        # selected_output_node = np.random.choice(self.output_neurons)
        # self.genes[innov_num] = Gene(innov_num, selected_input_node, selected_output_node)
                # print(input_neuron,output_neuron)
        # self.edges.append((selected_input_node.id, selected_output_node.id, 1.))

        input_nodes_id = [x.id for x in self.input_neurons]
        output_nodes_id = [x.id for x in self.output_neurons]


        # self.networkx_network = nx.from_edgelist(self.edges)
        self.networkx_network = nx.DiGraph()
        # self.networkx_network.add_edges_from(self.edges)
        # self.networkx_network.add_nodes_from(input_nodes_id)
        # self.networkx_network.add_nodes_from(output_nodes_id)
        self.networkx_network.add_weighted_edges_from(self.edges)

        self.fitness = 0.

        self.neuron_matrix = np.full((2,3), 0.03)
# a.species[0].find_all_routes(a.species[0].genomes[0])

        # network stuff
        # self.flatten = nn.Flatten()
    
        self.sparsity = (self.num_input_neurons+len(self.hidden_neurons))*self.num_output_neurons 
    def clone(self):
        return deepcopy(self)
    # def reset(self):
    #     self.model.reset()

    def set_fitness(self, fitness):
        self.fitness = fitness


    def set_generation(self, gen_id):
        self.generation_number = gen_id


    def set_species(self, s_id):
        self.species_number = s_id


    def get_next_neuron_id(self):
        current_id = self.current_neuron_id
        self.current_neuron_id += 1
        return current_id


    def reinitialize(self):
        for g_id, gene in self.genes.items():
            gene.randomize_weight()

    # def create_network_from_genes(self):

        #find how many 'layers' present between output and input layer
        
        #find genes that have output nodes
        # genes_connected_to_output_node = [5,9]
        #which nodes are next?
        # find distance next nodes to input layer


        # find the order of nodes for weights, which connected nodes are highest in order

    def is_compatible(self, comparison_genome):
        normalization_const = max(len(self.genes), len(comparison_genome.genes))
        normalization_const = normalization_const if normalization_const > 20 else 1
        
        num_excess_genes = len(self.get_excess_genes(comparison_genome))
        num_disjoint_genes = len(self.get_disjoint_genes(comparison_genome))
        avg_weight_diff = self.get_avg_weight_difference(comparison_genome)
        compatibility_score = ((num_excess_genes * config.EXCESS_COMPATIBILITY_CONSTANT) /
                                    normalization_const) +\
                              ((num_disjoint_genes * config.DISJOINT_COMPATIBILITY_CONSTANT) /
                                    normalization_const) +\
                              (avg_weight_diff * config.WEIGHT_COMPATIBILITY_CONSTANT)
        # print(compatibility_score)
        compatible = compatibility_score < config.COMPATIBILITY_THRESHOLD
        print(num_excess_genes,num_disjoint_genes,avg_weight_diff, compatibility_score)
        return compatible


    def get_excess_genes(self, comparison_genome):
        excess_genes = []
        largest_innovation_id = max(self.genes.keys())

        for g_id, gene in comparison_genome.genes.items():
            if g_id > largest_innovation_id:
                excess_genes.append(gene)

        return excess_genes


    def get_disjoint_genes(self, comparison_genome):
        disjoint_genes = []
        largest_innovation_id = max(self.genes.keys())
        # for g_id, genome in comparison_genome.genes.items():
        #     if not g_id in list(self.genes.values()) and g_id < largest_innovation_id:
        #         disjoint_genes.append(genome)

        # for g_id, genome in self.genes.items():
        #     if not g_id in list(comparison_genome.genes.values()):
        #         disjoint_genes.append(genome)

        # return disjoint_genes

        for g_id, genome in comparison_genome.genes.items():
            if not g_id in [x[0] for x in list(self.genes.items())] and g_id < largest_innovation_id:
                disjoint_genes.append(genome)

        for g_id, genome in self.genes.items():
            if not g_id in [x[0] for x in list(comparison_genome.genes.items())]:
                disjoint_genes.append(genome)

        return disjoint_genes


    def get_avg_weight_difference(self, comparison_genome):
        avg_weight_self = sum(gene.weight for gene in self.genes.values()) / len(self.genes)
        avg_weight_comp = sum(gene.weight for gene in comparison_genome.genes.values()) / len(comparison_genome.genes)
        return abs(avg_weight_self - avg_weight_comp)

    def calculate_sparsity(self):

        return len(self.genes)/((self.num_input_neurons + len(self.hidden_neurons))*self.num_output_neurons + \
            (self.num_input_neurons + len(self.hidden_neurons)-1)*(self.num_input_neurons + len(self.hidden_neurons)) / 2.)
    
    def mutate(self):
        # Genome Weight Mutations
        for gene in self.genes.values():
            gene.mutate_weight()
        
        for neuron in self.neurons.values():
            neuron.mutate_decay()
            neuron.mutate_threshold()

        # self.networkx_network = nx.DiGraph()
        # # self.networkx_network.add_edges_from(self.edges)
        # self.edges = [(x.input_neuron.id, x.output_neuron.id, 1.) if x.enabled else None for x in list(self.genes.values())]
        # self.edges = [x for x in self.edges if x is not None]
        # input_node_ids = [x.id for x in self.input_neurons]
        # output_node_ids = [x.id for x in self.output_neurons]
        # self.networkx_network.add_nodes_from(input_node_ids)
        # self.networkx_network.add_nodes_from(output_node_ids)
        # if self.hidden_neurons:
        #     self.networkx_network.add_nodes_from([x.id for x in self.hidden_neurons])
        # self.networkx_network.add_weighted_edges_from(self.edges)


        # Genome Structural Mutations
        # Adding Gene
        if np.random.uniform() < config.ADD_GENE_MUTATION:

            gene_added = False
            while not gene_added:

                # No valid genes exist to be added. Pay attention
                if (len(self.hidden_neurons) == 0):
                    break

                # Certain genes are not valid, such as any gene going to an input node, or any gene from an output node
                selected_input_node = np.random.choice(list(set().union(self.hidden_neurons, self.input_neurons)))
                selected_output_node = np.random.choice(list(set().union(self.hidden_neurons, self.output_neurons)))


                # neurons_lst_order = list(set().union(self.hidden_neurons, self.output_neurons))
                ####### function for implementing
                

                # add for probability of genes
                # neurons_lst_order.remove(selected_output_node)
                # # function that calculates disctance between selected input node and other nodes?
                # p_lst = []
                
                # for neuron in neurons_lst_order:
                #     neuron_id = neuron.id
                #     # if neuron_id==selected_input_node.id:
                #     #     continue
                #     print('go', selected_input_node.x_position,self.neurons[neuron_id].x_position )
                #     distance = np.sqrt((selected_input_node.x_position - self.neurons[neuron_id].x_position)**2 + (selected_input_node.y_position - self.neurons[neuron_id].y_position)**2)
                #     p_lst.append(distance)
                
                # p_array = np.asanyarray(p_lst)
                # p_array /= p_array.max()
                # p_array = 1. - p_array
                # p_array /= p_array.sum()
                
                # p_array[np.isnan(p_array)] = 0
                # selected_output_node = np.random.choice(neurons_lst_order, p=p_array)



                gene_valid = True
                for gene in self.genes.values():
                    # If this connection already exists, do not make the new gene
                    if (gene.input_neuron.id == selected_input_node.id and
                        gene.output_neuron.id == selected_output_node.id):
                        gene_valid = False
                        break

                # Can't have loops. Can't connect to itself. Gene must connect node from backwards to forwards. changed sign >=
                if (selected_input_node.id == selected_output_node.id):
                    gene_valid = False
#
                #can't be recurrent
                if nx.has_path(self.networkx_network, selected_output_node.id, selected_input_node.id):
                    gene_valid = False

                if gene_valid:
                    new_gene = Gene(self.innovation.get_new_innovation_number(),
                                    selected_input_node,
                                    selected_output_node)

                    self.networkx_network.add_edge(selected_input_node.id, selected_output_node.id, weight=-1.)
                    self.genes[new_gene.innovation_number] = new_gene
                    gene_added = True

        if np.random.uniform() < config.ADD_NODE_MUTATION:

            
            # Select gene at random and disable
            selected_gene = np.random.choice(list(self.genes.values()))
            
            #see if too many layers already exist 
            # density very hard to investigate 

            # Avoid adding the same neuron connection by not choosing a di.
            if selected_gene.enabled:
                selected_gene.disable()
                #should go somewhere else remake graph everytime it passes through here
                # self.networkx_network.remove_edge(selected_gene.input_neuron.id, selected_gene.output_neuron.id)

                # Create new node, rearrange ids to make higher neuron ids farther towards output layer
                # new_neuron = Neuron(selected_gene.output_neuron.id)
                # self.neurons[selected_gene.output_neuron.id] = new_neuron
                # selected_gene.output_neuron.set_id(self.get_next_neuron_id())
                # self.neurons[selected_gene.output_neuron.id] = selected_gene.output_neuron

                new_neuron = Neuron(selected_gene.output_neuron.id)

                # newly added for position
                print('goo', self.neurons[selected_gene.input_neuron.id].x_position, self.neurons[selected_gene.output_neuron.id].x_position)
                new_neuron.x_position = self.neurons[selected_gene.input_neuron.id].x_position + (self.neurons[selected_gene.output_neuron.id].x_position - self.neurons[selected_gene.input_neuron.id].x_position)/2.
                new_neuron.y_position = self.neurons[selected_gene.input_neuron.id].y_position + (self.neurons[selected_gene.output_neuron.id].y_position - self.neurons[selected_gene.input_neuron.id].y_position)/2.
                
                
                # self.neurons[selected_gene.output_neuron.id] = selected_gene.output_neuron
                selected_gene.output_neuron.set_id(self.get_next_neuron_id())
                self.neurons[selected_gene.output_neuron.id] = new_neuron

                

                # Create new genes
                new_input_gene = Gene(self.innovation.get_new_innovation_number(),
                                      selected_gene.input_neuron,
                                      new_neuron,
                                      1)
                self.networkx_network.add_edge(selected_gene.input_neuron.id, new_neuron.id, weight=-1.)

                new_output_gene = Gene(self.innovation.get_new_innovation_number(),
                                       new_neuron,
                                       selected_gene.output_neuron,
                                       selected_gene.weight)
                self.networkx_network.add_edge(new_neuron.id, selected_gene.output_neuron.id, weight=-1.)
                # Add to network
                self.genes[new_input_gene.innovation_number] = new_input_gene
                self.genes[new_output_gene.innovation_number] = new_output_gene
                self.hidden_neurons.append(new_neuron)

        # if np.random.uniform() < config.GENE_REMOVE_RATE:

        #     # Select gene at random and disable
        #     selected_gene = np.random.choice(list(self.genes.values()))
            
        #     # Avoid adding the same neuron connection by not choosing a di.
        
        #     output_neuron_input_genes_len = [x if x.enabled==True else None for x in selected_gene.output_neuron.input_genes.values()]
        #     output_neuron_input_genes_len = len([x for x in output_neuron_input_genes_len if x is not None])
        #     input_neuron_output_genes_len = [x if x.enabled==True else None for x in selected_gene.input_neuron.output_genes.values()]
        #     input_neuron_output_genes_len = len([x for x in input_neuron_output_genes_len if x is not None])

        #     if output_neuron_input_genes_len>=2 and input_neuron_output_genes_len>=2 and selected_gene.enabled:
        #         selected_gene.disable()

        #         print('GENE DISABLED')

        # rebuild network here:
        self.networkx_network = nx.DiGraph()
        # self.networkx_network.add_edges_from(self.edges)
        self.edges = [(x.input_neuron.id, x.output_neuron.id, 1.) if x.enabled else None for x in list(self.genes.values())]
        self.edges = [x for x in self.edges if x is not None]
        input_node_ids = [x.id for x in self.input_neurons]
        output_node_ids = [x.id for x in self.output_neurons]
        self.networkx_network.add_nodes_from(input_node_ids)
        self.networkx_network.add_nodes_from(output_node_ids)
        if self.hidden_neurons:
            self.networkx_network.add_nodes_from([x.id for x in self.hidden_neurons])
        self.networkx_network.add_weighted_edges_from(self.edges)
        self.sparsity = self.calculate_sparsity()
    # def find_node_snake(self, next_nodes):


    def mutate_hidden_layers_condition(self):
        

        for neuron in self.neurons.values():
            neuron.mutate_decay()
            neuron.mutate_threshold()


        if np.random.uniform() < config.ADD_GENE_MUTATION:

            gene_added = False
            while not gene_added:

                # No valid genes exist to be added. Pay attention
                if (len(self.hidden_neurons) == 0):
                    break

                # Certain genes are not valid, such as any gene going to an input node, or any gene from an output node
                selected_input_node = np.random.choice(list(set().union(self.hidden_neurons, self.input_neurons)))
                selected_output_node = np.random.choice(list(set().union(self.hidden_neurons, self.output_neurons)))

                gene_valid = True
                for gene in self.genes.values():
                    # If this connection already exists, do not make the new gene
                    if (gene.input_neuron.id == selected_input_node.id and
                        gene.output_neuron.id == selected_output_node.id):
                        gene_valid = False
                        # break

                # Can't have loops. Can't connect to itself. Gene must connect node from backwards to forwards. changed sign >=
                if (selected_input_node.id == selected_output_node.id):
                    gene_valid = False

#
                #can't be recurrent
                if nx.has_path(self.networkx_network, selected_output_node.id, selected_input_node.id):
                    gene_valid = False

                if gene_valid:
                    new_gene = Gene(self.innovation.get_new_innovation_number(),
                                    selected_input_node,
                                    selected_output_node)

                    self.networkx_network.add_edge(selected_input_node.id, selected_output_node.id, weight=-1.)
                    self.genes[new_gene.innovation_number] = new_gene
                    self.networkx_network = nx.DiGraph()
                    # self.networkx_network.add_edges_from(self.edges)
                    self.edges = [(x.input_neuron.id, x.output_neuron.id, 1.) if x.enabled else None for x in list(self.genes.values())]
                    self.edges = [x for x in self.edges if x is not None]
                    input_node_ids = [x.id for x in self.input_neurons]
                    output_node_ids = [x.id for x in self.output_neurons]
                    self.networkx_network.add_nodes_from(input_node_ids)
                    self.networkx_network.add_nodes_from(output_node_ids)
                    if self.hidden_neurons:
                        self.networkx_network.add_nodes_from([x.id for x in self.hidden_neurons])
                    self.networkx_network.add_weighted_edges_from(self.edges)
                    self.sparsity = self.calculate_sparsity()
                    gene_added = True
        
        

        if np.random.uniform() < config.ADD_NODE_MUTATION:

            
            # Select gene at random and disable
            selected_gene = np.random.choice(list(self.genes.values()))
            
        
            if selected_gene.enabled:
                selected_gene.disable()

                 # Create new node, rearrange ids to make higher neuron ids farther towards output layer
                # new_neuron = Neuron(selected_gene.output_neuron.id)
                # self.neurons[selected_gene.output_neuron.id] = new_neuron
                # selected_gene.output_neuron.set_id(self.get_next_neuron_id())
                # self.neurons[selected_gene.output_neuron.id] = selected_gene.output_neuron


                new_neuron = Neuron(selected_gene.output_neuron.id)

                # newly added for position
                # print('goo', self.neurons[selected_gene.input_neuron.id].x_position, self.neurons[selected_gene.output_neuron.id].x_position)
                new_neuron.x_position = self.neurons[selected_gene.input_neuron.id].x_position + (self.neurons[selected_gene.output_neuron.id].x_position - self.neurons[selected_gene.input_neuron.id].x_position)/2.
                new_neuron.y_position = self.neurons[selected_gene.input_neuron.id].y_position + (self.neurons[selected_gene.output_neuron.id].y_position - self.neurons[selected_gene.input_neuron.id].y_position)/2.
                
                
                # self.neurons[selected_gene.output_neuron.id] = selected_gene.output_neuron
                selected_gene.output_neuron.set_id(self.get_next_neuron_id())
                self.neurons[selected_gene.output_neuron.id] = new_neuron

                

                # Create new genes
                new_input_gene = Gene(self.innovation.get_new_innovation_number(),
                                    selected_gene.input_neuron,
                                    new_neuron,
                                    1)
                self.networkx_network.add_edge(selected_gene.input_neuron.id, new_neuron.id, weight=-1.)

                new_output_gene = Gene(self.innovation.get_new_innovation_number(),
                                    new_neuron,
                                    selected_gene.output_neuron,
                                    selected_gene.weight)
                self.networkx_network.add_edge(new_neuron.id, selected_gene.output_neuron.id, weight=-1.)
                # Add to network
                self.genes[new_input_gene.innovation_number] = new_input_gene
                self.genes[new_output_gene.innovation_number] = new_output_gene
                self.hidden_neurons.append(new_neuron)


                self.networkx_network = nx.DiGraph()
                # self.networkx_network.add_edges_from(self.edges)
                self.edges = [(x.input_neuron.id, x.output_neuron.id, 1.) if x.enabled else None for x in list(self.genes.values())]
                self.edges = [x for x in self.edges if x is not None]
                input_node_ids = [x.id for x in self.input_neurons]
                output_node_ids = [x.id for x in self.output_neurons]
                self.networkx_network.add_nodes_from(input_node_ids)
                self.networkx_network.add_nodes_from(output_node_ids)
                if self.hidden_neurons:
                    self.networkx_network.add_nodes_from([x.id for x in self.hidden_neurons])
                self.networkx_network.add_weighted_edges_from(self.edges)
                self.sparsity = self.calculate_sparsity()
    # def find_node_snake(self, next_nodes):
    # def find_all_routes(self):
        
    #     input_neurons_lst = [x.id for x in self.input_neurons]
    #     output_neurons_lst = [x.id for x in self.output_neurons]

    #     all_routes_lst = []
    #     for input_neuron in input_neurons_lst:
    #         for output_neuron in output_neurons_lst:
    #             # try:
    #             paths = list(nx.all_simple_paths(self.networkx_network, input_neuron, output_neuron))
    #             #neuron.id already extracted
    #             # paths = find_all_paths(genome.networkx_network, input_neuron, output_neuron)
    #             all_routes_lst.extend(paths)
                
    #             # except nx.NodeNotFound:
    #                 # print("not found, working though")

    #     longest_route = max(all_routes_lst, key=len)
    #     all_routes_lst.remove(longest_route)

    #     longest_routes = []
    #     longest_routes.append(longest_route)
    #     #see if multiple lists are of the same length
    #     if all_routes_lst:
    #         next_longest_route = max(all_routes_lst, key=len)
    #         while len(longest_routes[0])==len(next_longest_route):
    #             longest_routes.append(next_longest_route)
    #             # print(all_routes_lst)
    #             all_routes_lst.remove(next_longest_route)
    #             if all_routes_lst:
    #                 next_longest_route = max(all_routes_lst, key=len)
    #             else: 
    #                 break
    #     # print(longest_routes)
            
    #     # maybe change 10 in the future
    #     # neuron_matrix = np.full((list(genome.neurons.keys())[-1], len(longest_routes[0])), 0, dtype=object)
    #     neuron_matrix = np.full((10000, len(longest_routes[0])), 0, dtype=object)
    #     # neuron_matrix = np.zeros((100, len(longest_routes[0])))
    #     neuron_matrix[0:len(input_neurons_lst), 0] = np.asanyarray(input_neurons_lst).T
    #     neuron_matrix[0:len(output_neurons_lst), -1] = np.asanyarray(output_neurons_lst).T
    #     for i in range(len(longest_routes)):
    #         neuron_matrix[0+i,1:-1] = np.asanyarray(longest_routes[i][1:-1])
    #     # print('here the matrix', neuron_matrix)
    #     neurons_placed = list(np.unique(neuron_matrix))

    #     hidden_neurons = [x.id for x in self.hidden_neurons]
    #     hidden_neurons = [x for x in hidden_neurons if x not in neurons_placed]
    #     # print(hidden_neurons)

    #     genes = [[x.input_neuron.id, x.output_neuron.id] if x.enabled==True else None for x in self.genes.values()]
    #     genes = [x for x in genes if x is not None]

    #     gene_logbook = {}
    #     gene_logbook_cycle = {}

    #     row_hidden_nodes = max((neuron_matrix==0).argmax(axis=0))+10
    #     # a full cycle must be complete in where the dictionary has not changed before picking one
    #     cycle_condition = True
    #     while cycle_condition:
    #         hidden_neurons_start = hidden_neurons
    #         # gene_logbook_cycle = gene_logbook
    #         # print('herkenbaar', hidden_neurons)
    #         for neuron in hidden_neurons_start:
    #             # if neuron not in gene_logbook.keys():
    #             gene_logbook[neuron] = []
    #             list_genes_with_neuron = [x for x in genes if neuron in x]

    #             # print(list_genes_with_neuron)
    #             for gene in list_genes_with_neuron:
    #                 # print('genelogbook', gene_logbook)
    #                 if gene[0] == neuron:
    #                     location = np.where(neuron_matrix==gene[1])
    #                     # print(location)
    #                     if location[0].size!=0: # check if this is valid.
    #                         possible_locations = list(np.arange(1,location[1][0]))
    #                         gene_logbook[neuron].append(possible_locations)
    #                     else:
    #                         # if neuron in gene_logbook.keys():
    #                         #     relative_positions = np.asarray(gene_logbook[neuron]) - 1
    #                         #     possible_locations = list(relative_positions)
    #                         if gene[1] in gene_logbook.keys():
    #                             # print('did get here though0')
    #                             relative_positions = np.asarray(gene_logbook[gene[1]]) - 1
    #                             # print(relative_positions)
    #                             possible_locations = list(relative_positions)
    #                             gene_logbook[neuron].append(possible_locations)
    #                 elif gene[1] == neuron:
    #                     location = np.where(neuron_matrix == gene[0])
    #                     if location[0].size!=0:
    #                         possible_locations = list(np.arange(location[1][0]+1,neuron_matrix.shape[1]-1))
    #                         gene_logbook[neuron].append(possible_locations)
    #                     else:
    #                         # if neuron in gene_logbook.keys():
    #                         #     relative_positions = np.asarray(gene_logbook[neuron]) + 1
    #                         #     possible_locations = list(relative_positions)
    #                         if gene[0] in gene_logbook.keys():
    #                             # print('did get here though1')
    #                             relative_positions = np.asarray(gene_logbook[gene[0]]) + 1
    #                             # print(relative_positions)
    #                             possible_locations = list(relative_positions)
    #                             gene_logbook[neuron].append(possible_locations)

    #             # print('b', gene_logbook)
    #             len_constraints = (len(gene_logbook[neuron])-1)
    #             gene_logbook[neuron] = [x for xs in gene_logbook[neuron] for x in xs]

    #             # print('b', gene_logbook)
    #             counts = Counter(gene_logbook[neuron])
    #             dupids = [x for x in gene_logbook[neuron] if counts[x] > len_constraints ]
    #             gene_logbook[neuron] = list(set(dupids))
    #             # print('a', dupids)

    #             if len(gene_logbook[neuron])==1:
    #                 neuron_matrix[row_hidden_nodes, gene_logbook[neuron][0]] = neuron
    #                 row_hidden_nodes += 1
    #                 hidden_neurons.remove(neuron)
    #                 # If you del the bloody key, it is also taken out of the copy one, in this case gene_logbook_cycle
    #                 # del gene_logbook[neuron]
    #             # print(gene_logbook_cycle.values(), gene_logbook.values())
    #         if list(gene_logbook_cycle.values()) == list(gene_logbook.values()): 
    #             cycle_condition = False
    #             # print('it works')
    #         else:
    #             gene_logbook_cycle = copy.deepcopy(gene_logbook)
    

    #     if hidden_neurons:
    #         for neuron in hidden_neurons:
    #             # print(gene_logbook[neuron][0], neuron, row_hidden_nodes)
    #             try:
    #                 neuron_matrix[row_hidden_nodes, gene_logbook[neuron][0]] = neuron
                    
    #                 row_hidden_nodes += 1
    #             except IndexError:
    #                 pass
    #     network_neurons = [x.id for x in self.neurons.values()]

    #     matrix_neurons = list(np.unique(neuron_matrix))

    #     if not all(elem in network_neurons  for elem in matrix_neurons[1:]):
    #         print('no luck oi')

    #     network_genes = [[x.input_neuron.id, x.output_neuron.id] if x.enabled==True else None for x in self.genes.values()]
    #     network_genes = [x for x in network_genes if x is not None]
    #     for gene in network_genes:
    #         left_pos = np.where(neuron_matrix==gene[0])[1][0]
    #         right_pos = np.where(neuron_matrix==gene[1])[1][0]
    #         difference = right_pos - left_pos
    #         if difference == 1:
    #             continue
    #         else: 
    #             # neuron matrix left_pos + 1 in matrix == gene[0]
    #             # original: eft_pos+1:left_pos+1+difference
    #             neuron_matrix[row_hidden_nodes, left_pos+1:left_pos+difference] = np.full((1, difference-1), gene[0])
    #             row_hidden_nodes += 1
    #     return neuron_matrix




class Network_model(nn.Module):
    def __init__(self, layer_lst):
        super(Network_model).__init__()


        self.modules = []
        for i, j in enumerate(layer_lst[:-1]):
            self.modules.append(nn.Linear(j, 1))
            if len(layer_lst)>2:
                self.modules.append(nn.Sigmoid())
            
        self.linear_relu_stack = nn.Sequential(*self.modules)


    def rebuild_model(self, layer_lst):
        for i, j in enumerate(layer_lst[:-1]):
            self.modules.append(nn.Linear(j, layer_lst[i+1]))
            if len(layer_lst)>2:
                print('fuck you')
                self.modules.append(nn.Sigmoid())

        self.linear_relu_stack = nn.Sequential(*self.modules)

    def forward(self, x):
        # x = self.flatten(x)
        # m = nn.Sigmoid()
        output = self.linear_relu_stack(x)
        return output


def model_parameter_as_vector(model, parameter):  # to initialize xmean

# select only weights
    weights_vector = []

    to_be_adjusted_weights_only = [key for key, value in model.state_dict().items() if parameter in key.lower()]
    for key in to_be_adjusted_weights_only:
        # for curr_weights in model.state_dict()[key]:
        # Calling detach() to remove the computational graph from the layer.
        # numpy() is called for converting the tensor into a NumPy array.
        curr_weights = model.state_dict()[key].detach().numpy()
        vector = np.reshape(curr_weights, newshape=(curr_weights.size))
        weights_vector.extend(vector)
    return np.array(weights_vector)

def model_parameter_as_dict(model, weights_vector, parameter):
    to_be_adjusted_weights_only = [key for key, value in model.state_dict().items() if parameter in key.lower()]
    weights_dict = model.state_dict()
    start = 0
    for key in weights_dict:
        if key in to_be_adjusted_weights_only:
            # Calling detach() to remove the computational graph from the layer.
            # numpy() is called for converting the tensor into a NumPy array.
            w_matrix = weights_dict[key].detach().cpu().numpy()
            layer_weights_shape = w_matrix.shape
            layer_weights_size = w_matrix.size

            layer_weights_vector = weights_vector[start:start + layer_weights_size] #gaat hier iets fouts
            layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
            weights_dict[key] = torch.from_numpy(layer_weights_matrix)

            start = start + layer_weights_size
    return weights_dict

def build_model_param(param, model, parameter):
    with torch.no_grad():
        # alter function to only adjust weights, set biases to 0
        weight_dict = model_parameter_as_dict(model, param, parameter)
        model.load_state_dict(weight_dict)
    return model
# class model(object):
#     def __init__(self, model):
#         self.model = model


#     def pred(self, X):
#         output_network = self.model(X)
#         print(output_network)
#         output = nn.Sigmoid(output_network.detach())
#         return output

# model = model(Network([5,1]))
# a = model.pred(torch.randn(5))
# print(a)


# model = Network([5,1])
# print(model)

# X = torch.rand(1,5)
# logits = model(X)
# # pred_probab = nn.Sigmoid(logits)
# # y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {logits.item()}")