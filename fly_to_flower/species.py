from tkinter import W
from weakref import ref
import numpy as np
from scipy.stats import expon
from network import Network
import config as config
from env_3d_var_D_setpoint import LandingEnv3D as Quadhover
import matplotlib.pyplot as plt
# import FlapPyBird.flappy as flpy
import os
# os.chdir(os.getcwd() + '/FlapPyBird/')
from collections import Counter
from snn_pytorch_mod_double_output import SNN
import torch
import networkx as nx
from network_viz import draw_net
import copy 
from uuid import uuid4
from CMAES_NEAT import CMA_ES, CMA_ES_single


# uuid4()
# UUID('4a59bf4e-1653-450b-bc7e-b862d6346daa')

class objective:
    def __init__(self, environment):
        # self.model = model
        # self.model = self.model.to(device=device)

        self.environment = environment
        

    def position_encoder(self, div_lst, prob_ref):
        #current encoder
        D_plus = bool(div_lst[-1][0]>0.) * div_lst[-1][0]
        D_min = bool(div_lst[-1][0]<0.) * div_lst[-1][0]

        D_delta_plus = bool(div_lst[-1][0]>div_lst[-2][0]) * (div_lst[-1][0] - div_lst[-2][0])
        D_delta_min = bool(div_lst[-1][0]<div_lst[-2][0]) * (div_lst[-1][0] - div_lst[-2][0])
####        
        ref_div_node = bool(np.random.uniform()>=(1-prob_ref))
####
        x = torch.tensor([D_plus, D_min, D_delta_plus, D_delta_min, ref_div_node])

        return x


    def spike_encoder_div(self, OF_lst, prob_ref_wx, prob_ref_div, prob_ref_wy):
        #current encoder
        ref_div_node_plus = bool(np.random.uniform()>=(1-prob_ref_div[0]))
        D_plus = bool(OF_lst[-1][2]>0.) * 1.
        D_min = bool(OF_lst[-1][2]<0.) * 1.

        D_delta_plus = bool(OF_lst[-1][2]>OF_lst[-2][2]) * 1.
        D_delta_min = bool(OF_lst[-1][2]<OF_lst[-2][2]) * 1.
####        
        ref_div_node_minus = bool(np.random.uniform()>=(1-prob_ref_div[1]))
####
        ref_wx_node_plus = bool(np.random.uniform()>=(1-prob_ref_wx[0]))
        wx_plus = bool(OF_lst[-1][0]>0.) * 1.
        wx_min = bool(OF_lst[-1][0]<0.) * 1.

        wx_delta_plus = bool(OF_lst[-1][0]>OF_lst[-2][0]) * 1.
        wx_delta_min = bool(OF_lst[-1][0]<OF_lst[-2][0]) * 1.
####        
        ref_wx_node_minus = bool(np.random.uniform()>=(1-prob_ref_wx[1]))


        ref_wy_node_plus = bool(np.random.uniform()>=(1-prob_ref_wy[0]))
        wy_plus = bool(OF_lst[-1][1]>0.) * 1.
        wy_min = bool(OF_lst[-1][1]<0.) * 1.

        wy_delta_plus = bool(OF_lst[-1][1]>OF_lst[-2][1]) * 1.
        wy_delta_min = bool(OF_lst[-1][1]<OF_lst[-2][1]) * 1.
####        
        ref_wy_node_minus = bool(np.random.uniform()>=(1-prob_ref_wy[1]))
####

        x = torch.tensor([ref_wx_node_plus, wx_plus, wx_min, wx_delta_plus, wx_delta_min, ref_wx_node_minus, ref_div_node_plus, D_plus, D_min, D_delta_plus, D_delta_min, ref_div_node_minus, ref_wy_node_plus, wy_plus, wy_min, wy_delta_plus, wy_delta_min, ref_wy_node_minus])

        return x
    def spike_decoder(self, spike_array):


        pitch = spike_array[0] - spike_array[1]
        thrust = spike_array[2] - spike_array[3]
        roll = spike_array[4] - spike_array[5]

        return (pitch, thrust, roll)
    def objective_function_NEAT(self, model, ref_div, ref_wx, ref_wy):  # add prob here

        steps=100000
        
        mav_model = model
        # self.environment.ref_div = prob_ref*2
        self.environment.ref_div = ref_div
        self.environment.ref_wx = ref_wx
        self.environment.ref_wy = ref_wy

        self.environment.reset()
        divs_lst = []
        ref_lst = []

        reward_cum = 0
        for step in range(steps):
            prob_x = self.environment.x_prob()
            prob_z = self.environment.z_prob()
            prob_y = self.environment.y_prob()
            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_x, prob_z, prob_y)


            
            array = mav_model(encoded_input.float()) 
            control_input = self.spike_decoder(array.detach().numpy())
            divs, reward, done, _, _ = self.environment.step(np.asarray([control_input[2], control_input[0], control_input[1]]))
            if done:
                break
            divs_lst.append(self.environment.state[2][0])
            ref_lst.append(self.environment.height_reward)
            # divs_lst.append(self.environment.thrust_tc)
        # plt.plot(divs_lst)
        # plt.plot(ref_lst)
        mav_model.reset()    
        reward_cum = self.environment.reward
        # print(reward_cum, self.environment.state[0], )
        return reward_cum
    
    def objective_function_CMAES(self, x, ref_div, ref_wx, ref_wy, genome):  # add prob here
        # for i in [1.0, 1.5]:
        # ref_div = i
        steps=100000
        
        tags = list({x[0]: x[1].weight for x in genome.genes.items()}.keys())
        # weights = list({x[0]: x[1].weight for x in genome.genes.items()}.values())
        weights = list(x)


        gene_ad = 0
        for gene in tags: 
            genome.genes[gene].weight = weights[gene_ad]
            gene_ad = gene_ad + 1

        mav_model = place_weights(genome.neuron_matrix, genome)
        # print(mav_model.state_dict())
        # mav_model = build_model(x, self.model)

        # self.environment.ref_div = prob_ref*2
        self.environment.ref_div = ref_div
        self.environment.ref_wx = ref_wx
        self.environment.ref_wy = ref_wy

        self.environment.reset()
        divs_lst = []
        ref_lst = []
        # print(mav_model.state_dict())

        reward_cum = 0
        for step in range(steps):
            prob_x = self.environment.x_prob()
            prob_z = self.environment.z_prob()
            prob_y = self.environment.y_prob()
            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_x, prob_z, prob_y)
            array = mav_model(encoded_input.float()) 
            # print('b',array)
            control_input = self.spike_decoder(array.detach().numpy())
            # print('a', encoded_input, control_input)           
            divs, reward, done, _, _ = self.environment.step(np.asarray([control_input[2], control_input[0], control_input[1]]))

            if done:
                break
            divs_lst.append(self.environment.state[2][0])
            ref_lst.append(self.environment.height_reward)
            # divs_lst.append(self.environment.thrust_tc)
        
        mav_model.reset()    

        # time.sleep(0.1)
        # plt.plot(divs_lst, c='#4287f5')
        # plt.plot(ref_lst, c='#f29b29')

        
        # plt.ylabel('height (m)')
        # plt.xlabel('timesteps (0.02s)')
        # plt.title('Divergence: '+ str(ref_div))
        # plt.plot(divs_lst)
        
        # figure(figsize=(8, 6), dpi=80)
        # plt.show()
        # plt.savefig('pres_1.png')
        # plt.show()
        reward_cum = self.environment.reward
        # print(reward_cum, self.environment.state[2])
        return reward_cum

    def objective_function_CMAES_single(self, x, genome, ref_div=0.2, ref_wx=0.2, ref_wy=0.2):
        
        steps=100000
        tags = list({x[0]: x[1].weight for x in genome.genes.items()}.keys())
        weights = list(x)
        gene_ad = 0
        for gene in tags: 
            genome.genes[gene].weight = weights[gene_ad]
            gene_ad = gene_ad + 1

        mav_model = place_weights(genome.neuron_matrix, genome)
        # print('donehere')  
        self.environment.ref_div = ref_div
        self.environment.ref_wx = ref_wx
        self.environment.ref_wy = ref_wy
        
        self.environment.reset()
        reward_cum = 0
        for step in range(steps):
            prob_x = self.environment.x_prob()
            prob_z = self.environment.z_prob()
            prob_y = self.environment.y_prob()
            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_x, prob_z, prob_y)
            array = mav_model(encoded_input.float()) 
            control_input = self.spike_decoder(array.detach().numpy())
            divs, reward, done, _, _ = self.environment.step(np.asarray([control_input[2], control_input[0], control_input[1]]))
            if done:
                break

        mav_model.reset()    
        reward_cum = self.environment.reward
        return reward_cum

def find_all_routes(genome):
        
    input_neurons_lst = [x.id for x in genome.input_neurons]
    output_neurons_lst = [x.id for x in genome.output_neurons]

    all_routes_lst = []
    for input_neuron in input_neurons_lst:
        for output_neuron in output_neurons_lst:
            # try:
            paths = list(nx.all_simple_paths(genome.networkx_network, input_neuron, output_neuron))
              #neuron.id already extracted
            # paths = find_all_paths(genome.networkx_network, input_neuron, output_neuron)
            all_routes_lst.extend(paths)
            
            # except nx.NodeNotFound:
                # print("not found, working though")

    longest_route = max(all_routes_lst, key=len)
    all_routes_lst.remove(longest_route)

    longest_routes = []
    longest_routes.append(longest_route)
    #see if multiple lists are of the same length
    if all_routes_lst:
        next_longest_route = max(all_routes_lst, key=len)
        while len(longest_routes[0])==len(next_longest_route):
            longest_routes.append(next_longest_route)
            # print(all_routes_lst)
            all_routes_lst.remove(next_longest_route)
            if all_routes_lst:
                next_longest_route = max(all_routes_lst, key=len)
            else: 
                break
    # print(longest_routes)
        
    # maybe change 10 in the future
    # neuron_matrix = np.full((list(genome.neurons.keys())[-1], len(longest_routes[0])), 0, dtype=object)
    neuron_matrix = np.full((10000, len(longest_routes[0])), 0, dtype=object)
    # neuron_matrix = np.zeros((100, len(longest_routes[0])))
    neuron_matrix[0:len(input_neurons_lst), 0] = np.asanyarray(input_neurons_lst).T
    neuron_matrix[0:len(output_neurons_lst), -1] = np.asanyarray(output_neurons_lst).T
    for i in range(len(longest_routes)):
        neuron_matrix[0+i,1:-1] = np.asanyarray(longest_routes[i][1:-1])
    # neuron_matrix = clean_array(neuron_matrix)
    # print('here the matrix', neuron_matrix)
    neurons_placed = list(np.unique(neuron_matrix))

    hidden_neurons = [x.id for x in genome.hidden_neurons]
    hidden_neurons = [x for x in hidden_neurons if x not in neurons_placed]
    # print(hidden_neurons)

    genes = [[x.input_neuron.id, x.output_neuron.id] if x.enabled==True else None for x in genome.genes.values()]
    genes = [x for x in genes if x is not None]

    gene_logbook = {}
    gene_logbook_cycle = {}

    row_hidden_nodes = max((neuron_matrix==0).argmax(axis=0))+100
    # a full cycle must be complete in where the dictionary has not changed before picking one
    cycle_condition = True
    # print(neuron_matrix)
    while cycle_condition:
        hidden_neurons_start = hidden_neurons
        # print(hidden_neurons_start)
        # gene_logbook_cycle = gene_logbook
        # print('herkenbaar', hidden_neurons)
        for neuron in hidden_neurons_start:
            # if neuron not in gene_logbook.keys():
            gene_logbook[neuron] = []
            list_genes_with_neuron = [x for x in genes if neuron in x]

            # print(list_genes_with_neuron)
            for gene in list_genes_with_neuron:
                # print('genelogbook', gene_logbook)
                if gene[0] == neuron:
                    location = np.where(neuron_matrix==gene[1])
                    # print(location)
                    if location[0].size!=0: # check if this is valid.
                        possible_locations = list(np.arange(1,location[1][0]))
                        # print('aa', gene[1], possible_locations)
                        gene_logbook[neuron].append(possible_locations)
                    else:
                        # if neuron in gene_logbook.keys():
                        #     relative_positions = np.asarray(gene_logbook[neuron]) - 1
                        #     possible_locations = list(relative_positions)
                        if gene[1] in gene_logbook.keys():
                            # print('did get here though0')
                            # relative_positions = np.asarray(gene_logbook[gene[1]]) - 1 #original
                            # print(gene_logbook[gene[1]][-1])
                            relative_positions = list(np.arange(1,gene_logbook[gene[1]][-1]))
                            # print('a', gene[1],relative_positions)
                            possible_locations = relative_positions
                            gene_logbook[neuron].append(possible_locations)
                elif gene[1] == neuron:
                    location = np.where(neuron_matrix == gene[0])
                    if location[0].size!=0:
                        possible_locations = list(np.arange(location[1][0]+1,neuron_matrix.shape[1]-1))
                        # print('bb', gene[0], possible_locations)
                        gene_logbook[neuron].append(possible_locations)
                    else:
                        # if neuron in gene_logbook.keys():
                        #     relative_positions = np.asarray(gene_logbook[neuron]) + 1
                        #     possible_locations = list(relative_positions)
                        if gene[0] in gene_logbook.keys():
                            # print((np.asarray(gene_logbook[gene[0]])+1)[0],neuron_matrix.shape[1]-1)
                            # relative_positions = np.asarray(gene_logbook[gene[0]]) + 1 # original
                            relative_positions = list((np.arange((np.asarray(gene_logbook[gene[0]])+1)[0],neuron_matrix.shape[1]-1)))
                            # print('b', gene[0], relative_positions)
                            possible_locations = relative_positions
                            gene_logbook[neuron].append(possible_locations)

            # print('b', gene_logbook)
            # print(gene_logbook)
            len_constraints = (len(gene_logbook[neuron])-1)
            gene_logbook[neuron] = [x for xs in gene_logbook[neuron] for x in xs]

            # print('b', gene_logbook)
            #als maar 1 kant constraints heeft, toch toevoegen, die van de andere kant is niet duidelijk


            counts = Counter(gene_logbook[neuron])
            dupids = [x for x in gene_logbook[neuron] if counts[x] > len_constraints ]
            # print('zzzzzzzz', list(set(dupids)), sorted(list(set(dupids))))
            gene_logbook[neuron] = sorted(list(set(dupids)))
            # print('a', dupids)

            if len(gene_logbook[neuron])==1:
                neuron_matrix[row_hidden_nodes, gene_logbook[neuron][0]] = neuron
                row_hidden_nodes += 1
                hidden_neurons.remove(neuron)
                # If you del the bloody key, it is also taken out of the copy one, in this case gene_logbook_cycle
                # del gene_logbook[neuron]
            # print(gene_logbook_cycle.values(), gene_logbook.values())
        if list(gene_logbook_cycle.values()) == list(gene_logbook.values()): 
            cycle_condition = False
            # print('it works')
        else:
            gene_logbook_cycle = copy.deepcopy(gene_logbook)
   
    # print(neuron_matrix, gene_logbook, hidden_neurons)
    if hidden_neurons:
        for neuron in hidden_neurons:
            # print(gene_logbook[neuron][0], neuron, row_hidden_nodes)
            try:
                # print(neuron, gene_logbook[neuron])
                neuron_matrix[row_hidden_nodes, gene_logbook[neuron][0]] = neuron
                
                row_hidden_nodes += 1
            except IndexError:
                print(neuron)
                pass
    # neuron_matrix = clean_array(neuron_matrix)
    network_neurons = [x.id for x in genome.neurons.values()]

    matrix_neurons = list(np.unique(neuron_matrix))

    if not all(elem in network_neurons  for elem in matrix_neurons[1:]):
        print('no luck oi')



    network_genes = [[x.input_neuron.id, x.output_neuron.id] if x.enabled==True else None for x in genome.genes.values()]
    network_genes = [x for x in network_genes if x is not None]
    for gene in network_genes:
        # print(neuron_matrix, gene[1], np.where(neuron_matrix==gene[1]))
        left_pos = np.where(neuron_matrix==gene[0])[1][0]
        right_pos = np.where(neuron_matrix==gene[1])[1][0]
        difference = right_pos - left_pos
        if difference == 1:
            continue
        else: 
            # neuron matrix left_pos + 1 in matrix == gene[0]
            # original: eft_pos+1:left_pos+1+difference
            neuron_matrix[row_hidden_nodes, left_pos+1:left_pos+difference] = np.full((1, difference-1), gene[0])
            row_hidden_nodes += 1
    # neuron_matrix = clean_array(neuron_matrix)
    # print(neuron_matrix)
    return neuron_matrix

def clean_array(array):
    for column in range(len(array[0,:])-2):
        u, indices = np.unique(array[:,column+1], return_index=True)
        # print(u[:-1])
        # print(u)
        u = u[u!=0]
        # print(np.asarray(u).reshape(-1,1))
        array[:len(u), column+1] = np.asarray(u)

        array[len(u):, column+1] = 0
    sumrow = np.abs(array).sum(-1)
    array = array[sumrow>0]
    return array



# find_all_routes(a.species[3].genomes[8])

def place_weights(neuron_matrix, genome):
    neurons_lst = []
    for column in range(neuron_matrix.shape[1]):
        row = neuron_matrix[:,column]
        row = row[row!=0]
        neurons_lst.append(len(row))
    
    network_genes = [[x.innovation_number, x.input_neuron.id, x.output_neuron.id, x.weight] if x.enabled==True else None for x in genome.genes.values()]
    network_genes = [x for x in network_genes if x is not None]

    gene_dct = {}
    gene_dct_weight = {}
    for geneset in network_genes:
        # print(geneset)
        gene_dct[geneset[0]] = (geneset[1], geneset[2])
        gene_dct_weight[geneset[0]] = geneset[3]

    
    # list(mydict.keys())[list(mydict.values()).index(16)]

    # print(list(gene_dct.keys()))
    model = SNN(neurons_lst[:-1], 6)
    # create function in model to return the name of the layers!
    # neuron's already passed :
    # all input neurons oi

    weights_dict = model.state_dict()

    # weights_dict_boolean = model.state_dict()
    # print(weights_dict)

    decay_layer = []

    threshold_layer = []

    for layer in range(len(neurons_lst)-1):
        right_nodes = neuron_matrix[:,layer+1]
        right_nodes = right_nodes[right_nodes!=0]
        left_nodes = neuron_matrix[:,layer]
        left_nodes = left_nodes[left_nodes!=0]
        # print(left_nodes, right_nodes)
        fill_array = np.full((len(right_nodes), len(left_nodes)), 0.0002)
        # fill_array_boolean = np.full((len(right_nodes), len(left_nodes)), False)
        for right_node_pos in range(len(right_nodes)):
            decay = False
            threshold = False
            for left_node_pos in range(len(left_nodes)):
                # print(right_nodes[right_node_pos], left_nodes[left_node_pos])
                # decay = 0.1
                # threshold = 1.0
                try:
                    key = (left_nodes[left_node_pos], right_nodes[right_node_pos])
                    innovation_number = list(gene_dct.keys())[list(gene_dct.values()).index(key)]
                    del gene_dct[innovation_number]
                    fill_array[right_node_pos, left_node_pos] = gene_dct_weight[innovation_number]
                    # fill_array_boolean[right_node_pos, left_node_pos] = True

                    decay = genome.neurons[right_nodes[right_node_pos]].v_decay
                    threshold = genome.neurons[right_nodes[right_node_pos]].threshold
                    # continue  # added this later, is this correct?
                    # print('iii', (left_nodes[left_node_pos], right_nodes[right_node_pos]) )
                except ValueError:
                    pass

                if str(right_nodes[right_node_pos])==str(left_nodes[left_node_pos]):
                    fill_array[right_node_pos, left_node_pos] = 1.
                    #doorpaas properties
                    # decay_layer.append(0.2)
                    # threshold_layer.append(0.3)
                    decay = 0.2
                    threshold = 0.3
                # elif str(left_nodes[left_node_pos]) + str(right_nodes[right_node_pos]) in list(gene_dct.keys()):
                else:
                    None
                    #non specifieke door paas, hoge decay, hoge threshold
                    # decay_layer.append(0.1)
                    # threshold_layer.append(1.0)
                    # decay = 0.01
                    # threshold = 1.0
            if not decay:
                decay = genome.neurons[right_nodes[right_node_pos]].v_decay
                threshold = genome.neurons[right_nodes[right_node_pos]].threshold

            decay_layer.append(decay)
            threshold_layer.append(threshold)
            # print('hjlk', layer)
        # it has to do with this
        if 'snn.synapses.' + str(layer) + '.weight' in weights_dict.keys():
            weights_dict['snn.synapses.' + str(layer) + '.weight'] = torch.from_numpy(fill_array)
            weights_dict['snn.neurons.' + str(layer) + '.v_decay'] = torch.from_numpy(np.asarray(decay_layer))
            weights_dict['snn.neurons.' + str(layer) + '.thresh'] = torch.from_numpy(np.asarray(threshold_layer))

            # weights_dict_boolean['snn.synapses.' + str(layer) + '.weight'] = torch.from_numpy(fill_array_boolean)

        elif layer==len(neurons_lst)-2:
            weights_dict['last_layer.synapses.0.weight'] = torch.from_numpy(fill_array)
            weights_dict['last_layer.neurons.0.v_decay'] = torch.from_numpy(np.asarray(decay_layer))

        else:
            print('someting wong')

        decay_layer = []
        threshold_layer = []
    model.load_state_dict(weights_dict)
    return model


class Species(object):


    def __init__(self, s_id, species_population, genome, genomes=None):
        self.species_id = s_id
        self.species_population = species_population
        self.generation_number = 0
        self.species_genome_representative = genome

        genome.set_species(self.species_id)
        genome.set_generation(self.generation_number)
        self.genomes = {i:genome.clone() for i in range(self.species_population)}
        for i in range(1, self.species_population):
            self.genomes[i].reinitialize()

        if genomes:
            self.genomes = genomes
        # Information used for culling and population control
        self.active = True
        self.no_improvement_generations_allowed = config.STAGNATED_SPECIES_THRESHOLD
        self.times_stagnated = 0
        self.avg_max_fitness_achieved = -100000000000.
        self.generation_with_max_fitness = 0


        #temporarily
        self.high_con = 3.

        self.best_genome = 0
        

    def run_generation(self, cycle, div_training, wx_training, wy_training, cma_es_learning):
        if self.active: #this should always be true unless species are allowed to die
            species_fitness = self.generate_fitness(cycle, div_training, wx_training, wy_training, cma_es_learning)
            # I don't particularly like this +1 fix... as it will skew populations
            avg_species_fitness = float(species_fitness)/float(self.species_population)
            self.culling(avg_species_fitness)
            return avg_species_fitness if self.active else None
        else:
            return None


    def evolve(self):
        if self.active:
            survivor_ids = self.select_survivors()
            print('got here')
            self.create_next_generation(survivor_ids)
            
            self.generation_number += 1
            for genome in self.genomes.values():
                genome.set_generation(self.generation_number)

    #     # save fitness for each genome
    #     # calculate average fitness of species EASY LETSGO


    
    # This function holds the interface and interaction with FlapPyBird
    # def generate_fitness(self):
    #     species_score = 0
    #     for i in self.genomes.keys():
    #         score = np.random.uniform(1.,self.high_con)
    #         # print(score)
    #         self.genomes[i].set_fitness(score)
    #         # self.high_con += 1.
    #         species_score += score

    #     return species_score




    def generate_fitness(self, cycle, div_training, wx_training, wy_training, cma_es_learning=False):
        species_score = 0
        genome_scores = {}
        for genome_id, genome in self.genomes.items():
            if self.genomes[genome_id].learnable:
                # if self.genomes[genome_id].learn:
                # draw_net(genome)
                # try:
                print('round: ', cycle, 'species:', self.species_id, '    genome', genome_id)
                neuron_matrix = find_all_routes(self.genomes[genome_id])
                neuron_matrix = clean_array(neuron_matrix)
                self.genomes[genome_id].neuron_matrix = neuron_matrix

                environment = Quadhover()
                objective_genome = objective(environment)

                
                # CMA-ES learning 
                if cma_es_learning:
                # if genome_id==0:
                
                    cycles = 5 + int(len(self.genomes[genome_id].hidden_neurons)/2.)
                    cycles = 2
                    tags = list({x[0]: x[1].weight for x in self.genomes[genome_id].genes.items()}.keys())
                    weights = np.asarray(list({x[0]: x[1].weight for x in self.genomes[genome_id].genes.items()}.values()))

                    # print('aii', weights)
                    # for cycle in range(cycles):  
                    cma_es_class  = CMA_ES(objective_genome.objective_function_CMAES, N=weights.shape[0], xmean=weights, genome=self.genomes[genome_id])
                    new_weights, best_fitness = cma_es_class.optimize_run(cycles, div_training, wx_training, wy_training, self.genomes[genome_id])
                    
                    # print('aai', new_weights)

                    gene_ad = 0
                    for gene in tags:
                        self.genomes[genome_id].genes[gene].weight = new_weights[gene_ad]
                        gene_ad = gene_ad + 1



                model = place_weights(neuron_matrix, self.genomes[genome_id])
                # break
                # environment = QuadHover()
                # objective_genome = objective(environment)
                        
                reward = 0
                for i in range(len(div_training)):
                    add = objective_genome.objective_function_NEAT(model, div_training[i], wx_training[i], wy_training[i]) + objective_genome.objective_function_NEAT(model, div_training[i], wx_training[i], wy_training[i])
                    reward = reward + add/2.
                    # print(div_training[i], wx_training[i], add)
                # reward = reward/float(len(div_training))
                print('reward', reward)
            
            else:
                for i in range(len(div_training)):
                    add = objective_genome.objective_function_NEAT(model, div_training[i], wx_training[i], wy_training[i]) + objective_genome.objective_function_NEAT(model, div_training[i], wx_training[i], wy_training[i])
                    reward = reward + add/2.
                    # print(div_training[i], wx_training[i], add)
                # reward = reward/float(len(div_training))
                print('reward for unlearnable', reward)


            self.genomes[genome_id].set_fitness(-reward)
            genome_scores[genome_id] = -reward
            species_score += -reward
        # print(genome_scores)
        self.best_genome = max(genome_scores, key=genome_scores.get)
        return species_score

    # def first_round_evolutionary_process_species(self, div_training, wx_training):
    #     for genome_id, genome in self.genomes.items():
    #         # draw_net(genome)
    #         # try:
    #         print('species:', self.species_id, '    genome', genome_id)
    #         neuron_matrix = find_all_routes(self.genomes[genome_id])
    #         neuron_matrix = clean_array(neuron_matrix)
    #         self.genomes[genome_id].neuron_matrix = neuron_matrix

    #         environment = Quadhover()
    #         objective_genome = objective(environment)
    #         model = place_weights(neuron_matrix, self.genomes[genome_id])
    #         reward = 0.
    #         for i in range(len(div_training)):
    #                 add = objective_genome.objective_function_NEAT(model, div_training[i], wx_training[i])  + objective_genome.objective_function_NEAT(model, div_training[i], wx_training[i])
    #                 reward = reward + add/2.
    #             # print(div_training[i], wx_training[i], add)
    #         # reward = reward/float(len(div_training))
    #         print('reward', reward)



    def create_next_generation(self, replicate_ids):
        genomes = {}

        # Champion of each species is copied to next generation unchanged
        genomes[0] = self.genomes[replicate_ids[0]].clone()


        genome_id = 1

        # Spawn a generation consisting of progeny from fittest predecessors


        while (genome_id < self.species_population):
            

            # Choose an old genome at random from the survivors
            

            index_choice = self.get_skewed_random_sample(len(replicate_ids))
            random_genome = self.genomes[replicate_ids[index_choice]].clone()

            # Clone
            if np.random.uniform() > config.CROSSOVER_CHANCE:
                genomes[genome_id] = random_genome

            # Crossover
            else:
                index_choice_mate = self.get_skewed_random_sample(len(replicate_ids))
                random_genome_mate = self.genomes[replicate_ids[index_choice_mate]].clone()                
                genomes[genome_id] = self.crossover(random_genome, random_genome_mate)

            # Mutate the newly added genome
            learning_condition = False
            
            temp_genome = genomes[genome_id].clone()
            while not learning_condition:
                temp_genome_layers = temp_genome.clone()
                for i in range(1):       
                    try:
                        temp_genome_layers.mutate()
                    except Exception as e:
                        temp_genome_layers = temp_genome.clone()
                
                try:
                    neuron_matrix = find_all_routes(temp_genome_layers)
                    if 6 >neuron_matrix.shape[1]:
                        temp_genome = temp_genome_layers.clone()
                        learning_condition = True
                    else:
                        # temp_genome_layers = temp_genome.clone()
                        learning_condition = False

                    
                except Exception as e:
                    learning_condition = False
                # neuron_matrix = find_all_routes(temp_genome)
                # neuron_matrix = clean_array(neuron_matrix)
                # temp_genome.neuron_matrix = neuron_matrix

                # environment = Quadhover()
                # objective_genome = objective(environment)

                # cycles = 3
                # weights = np.asarray(list({x[0]: x[1].weight for x in temp_genome.genes.items()}.values()))

                # cma_es_class  = CMA_ES_single(objective_genome.objective_function_CMAES_single, N=weights.shape[0], xmean=weights, genome=temp_genome)
                # new_weights, best_fitness, condition = cma_es_class.optimize_run(cycles, 0.5, 0.5, 0.5, temp_genome)
                # if condition:
                    # learning_condition = True
                    # temp_genome.learnable = True
                    # print('passed test', genome_id)
                # else:
                #     temp_genome.learnable = False
                #     learning_condition = True


                #mutate only threshold and decay until fits constraints here
                # except Exception as e:
                #     learning_condition = False
            genomes[genome_id] = temp_genome.clone()
            genome_id += 1

        self.genomes = genomes
        
    def create_random_network(self, genome_id, hidden_neurons, hidden_layers):
        temp_genome = self.genomes[genome_id].clone()
        learning_condition = False
        

        while not learning_condition:
            temp_genome_layers = temp_genome.clone()
            while len(temp_genome.hidden_neurons)<hidden_neurons: 
                try:
                    temp_genome_layers.mutate_hidden_layers_condition()
                    neuron_matrix = find_all_routes(temp_genome_layers)
                    neuron_matrix = clean_array(neuron_matrix)
                    # print(neuron_matrix.shape, neuron_matrix)
                    temp_genome_layers.neuron_matrix = neuron_matrix
                except Exception as e:
                    temp_genome_layers = temp_genome.clone()
                    print('gonewrongyo')
                if hidden_layers >neuron_matrix.shape[1]:
                    temp_genome = temp_genome_layers.clone()
                else:
                    temp_genome_layers = temp_genome.clone()
                    
            try:
                neuron_matrix = find_all_routes(temp_genome)
                neuron_matrix = clean_array(neuron_matrix)
                print(neuron_matrix.shape)
                temp_genome.neuron_matrix = neuron_matrix

              
                learning_condition = True
                
            except Exception as e:
                    temp_genome = self.genomes[genome_id].clone()


        # while not learning_condition:
        #     temp_genome_layers = temp_genome.clone()
        #     while len(temp_genome.hidden_neurons)<hidden_neurons: 
        #         try:
        #             temp_genome_layers.mutate_hidden_layers_condition()
        #             neuron_matrix = find_all_routes(temp_genome_layers)
        #             neuron_matrix = clean_array(neuron_matrix)
        #             # print(neuron_matrix.shape, neuron_matrix)
        #             temp_genome_layers.neuron_matrix = neuron_matrix
        #         except Exception as e:
        #             temp_genome_layers = temp_genome.clone()
        #             print('gonewrongyo')
        #         if hidden_layers >neuron_matrix.shape[1]:
        #             temp_genome = temp_genome_layers.clone()
        #         else:
        #             temp_genome_layers = temp_genome.clone()
                    
        #     try:
        #         neuron_matrix = find_all_routes(temp_genome)
        #         neuron_matrix = clean_array(neuron_matrix)
        #         print(neuron_matrix.shape)
        #         temp_genome.neuron_matrix = neuron_matrix

        #         environment = Quadhover()
        #         objective_genome = objective(environment)
                
        #         cycles = 5
        #         weights = np.asarray(list({x[0]: x[1].weight for x in temp_genome.genes.items()}.values()))

        #         cma_es_class  = CMA_ES_single(objective_genome.objective_function_CMAES_single, N=weights.shape[0], xmean=weights, genome=temp_genome)
        #         new_weights, best_fitness, condition = cma_es_class.optimize_run(cycles, 0.5, 0.5, 0.5, temp_genome)
        #         print('here noww')
        #         if condition:

        #             learning_condition = True
        #             print('passed test', genome_id)
        #         else:
        #             temp_genome = self.genomes[genome_id].clone()
        #     except Exception as e:
        #             temp_genome = self.genomes[genome_id].clone()
        

        self.genomes[genome_id] = temp_genome.clone()



        # network is too big!!!!!!!
        # if neuron matrix more than blabla rows, don't add neuron

    def crossover(self, random_genome, random_genome_mate):
        if random_genome.fitness > random_genome_mate.fitness:
            fit_genome, unfit_genome = random_genome, random_genome_mate
        else:
            fit_genome, unfit_genome = random_genome_mate, random_genome

        for g_id, gene in fit_genome.genes.items():

            # If it has key, it is a matching gene 
            # unfit_genome.genes.has_key(g_id)
            if g_id in unfit_genome.genes:

                # Randomly inherit from unfit genome
                if np.random.uniform(-1., 1.) < -0.5:
                    gene.weight = unfit_genome.genes[g_id].weight

                # Have chance of disabling if either parent is disabled
                if not gene.enabled or not unfit_genome.genes[g_id].enabled:
                    if np.random.uniform() < config.INHERIT_DISABLED_GENE_RATE:
                        if not (gene.input_neuron.id in fit_genome.networkx_network.nodes) and not (gene.output_neuron.id in fit_genome.networkx_network.nodes):
                            gene.disable()
                        else:
                            None

        for n_id, neuron in fit_genome.neurons.items():

            if n_id in unfit_genome.neurons:
                if np.random.uniform(-1., 1.) < 0:
                    neuron.v_decay = unfit_genome.neurons[n_id].v_decay
                    neuron.threshold = unfit_genome.neurons[n_id].threshold


        
        return fit_genome


    def select_survivors(self):
        sorted_network_ids = sorted(self.genomes, 
                                    key=lambda k: self.genomes[k].fitness,
                                    reverse=True)
        print('sorted_network_ids', sorted_network_ids, [x.fitness for x in self.genomes.values()])
        # alive_network_ids = sorted_network_ids[:int(round(float(self.species_population)*0.5))]
        alive_network_ids = sorted_network_ids[:int(round(float(self.species_population)*0.5))]
        # dead_network_ids = sorted_network_ids[int(round(float(self.species_population)/2.0)):]

        return alive_network_ids


    def culling(self, new_avg_fitness):
        if new_avg_fitness > self.avg_max_fitness_achieved:
            self.avg_max_fitness_achieved = new_avg_fitness
            self.generation_with_max_fitness = self.generation_number
        
        # Cull/Repopulate due to stagnation 
        if (self.generation_number - self.generation_with_max_fitness) > self.no_improvement_generations_allowed:
            self.times_stagnated += 1
            if self.times_stagnated > config.STAGNATIONS_ALLOWED:
                print("Species", self.species_id, "culled due to multiple stagnations.")
                self.active = False
            else:
                print("Species", self.species_id, "stagnated. Repopulating...")
                self.generation_with_max_fitness = self.generation_number
                self.avg_max_fitness_achieved = -100000000000.
                # Get a random genome (just to maintain the structure)  $$$$$$ why not use own gnome?
                genome = self.genomes[0]
                self.genomes = {i:genome.clone() for i in range(self.species_population)}
                # Reinitialize, otherwise if we just clone the champion, we may end up with same local optima
                for genome in self.genomes.values():
                    genome.reinitialize()

        # Cull due to weak species
        if (self.species_population < config.WEAK_SPECIES_THRESHOLD):
            print("Species", self.species_id, "culled due to lack of breeding resulting in low population.")
            self.active = False


    def add_genome(self, genome):
        genome.set_species(self.species_id)
        genome.set_generation(self.generation_number)
        self.genomes[self.species_population] = genome.clone()
        self.species_population += 1


    def delete_genome(self, genome_id):
        self.genomes[genome_id] = self.genomes[self.species_population-1].clone()
        del self.genomes[self.species_population-1]
        self.species_population -= 1


    def set_population(self, population):
        self.species_population = population


    def increment_population(self, population_change=1):
        self.species_population += population_change


    def decrement_population(self, population_change=1):
        # print('here')
        self.species_population -= population_change


    def get_skewed_random_sample(self, n, slope=-1.0):
        """
        Randomly choose an index from an array of some given size using a scaled inverse exponential 

        n: length of array
        slope: (float) determines steepness of the probability distribution
               -1.0 by default for slightly uniform probabilities skewed towards the left
               < -1.0 makes it more steep and > -1.0 makes it flatter
               slope = -n generates an approximately uniform distribution
        """
        inv_l = 1.0/(n**float(slope)) # 1/lambda
        x = np.array([i for i in range(0,n)]) # list of indices 

        # generate inverse exponential distribution using the indices and the inverse of lambda
        p = expon.pdf(x, scale=inv_l)

        # generate uniformly distributed random number and weigh it by total sum of pdf from above
        rand = np.random.random() * np.sum(p)
        
        for i, p_i in enumerate(p):
            # chooses an index by checking whether the generated number falls into a region around 
            # that index's probability, where the region is sized based on that index's probability 
            rand -= p_i
            if rand < 0:
                return i

        return 0


    def pretty_print_s_id(self, s_id):
        # print "\n"
        # print "===================="
        # print "===  Species:", s_id, " ==="
        # print "===================="
        # print "\n"
        print(str(s_id))


    def pretty_print_gen_id(self, gen_id):
        # print "-----------------------"
        # print "---  Generation:", gen_id, " ---"
        # print "-----------------------"
        # print "\n"
        print(str(gen_id))

