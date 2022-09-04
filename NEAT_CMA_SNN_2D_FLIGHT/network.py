import torch 
import torch.nn as nn
from copy import deepcopy
from neuron import Neuron
from gene import Gene
import numpy as np
import config as config


import networkx as nx
# from collections import namedtuple
# import numpy as np
# from typing import Optional, NamedTuple, Tuple, Any, Sequence

# class SpikeFunction(torch.autograd.Function):
#     """
#     Spiking function with rectangular gradient.
#     Source: https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full
#     Implementation: https://github.com/combra-lab/pop-spiking-deep-rl/blob/main/popsan_drl/popsan_td3/popsan.py
#     """

#     @staticmethod
#     def forward(ctx, v):
#         ctx.save_for_backward(v)  # save voltage - thresh for backwards pass
#         return v.gt(0.0).float()

#     @staticmethod
#     def backward(ctx, grad_output):
#         v, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         spike_pseudo_grad = (v.abs() < 0.5).float()  # 0.5 is the width of the rectangle
#         return grad_input * spike_pseudo_grad, None  # ensure a tuple is returned

# #cant call numpy on tensor thatrequires grad. use tensor.detach().numpy()

# # class SpikeFunction(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #     def forward(v):
# #         return v.gt(0.0).float()


# # # Placeholder for LIF state
# LIFState = namedtuple('LIFState', ['z', 'v'])


# class LIF(nn.Module):
#     """
#     Leaky-integrate-and-fire neuron with learnable parameters.
#     """

#     def __init__(self, size):
#         super().__init__()
#         self.size = size
#         # Initialize all parameters randomly as U(0, 1)
#         # self.v_decay = nn.Parameter(torch.rand(size))
#         self.v_decay = nn.Parameter(torch.FloatTensor(size).uniform_(0.3, 0.85)) #here
#         # self.thresh = nn.Parameter(torch.rand(size))
#         self.thresh = nn.Parameter(torch.FloatTensor(size).uniform_(0.7, 0.999))  #here
#         self.spike = SpikeFunction.apply  # spike function

#     def forward(self, synapse, z, state = None):
#         # Previous state
#         if state is None:
#             state = LIFState(
#                 z=torch.zeros_like(synapse(z)),
#                 v=torch.zeros_like(synapse(z)),
#             )
#         # Update state
#         i = synapse(z)
#         v = state.v * self.v_decay * (1.0 - state.z) + i
#         z = self.spike(v - self.thresh)
        
#         return z, LIFState(z, v)

# ###########
# class LIFState_no_thresh(NamedTuple):
#     v: torch.Tensor

# # Placeholder for LIF parameters
# class LIFParameters_no_tresh(NamedTuple):
#     v_decay: torch.Tensor = torch.as_tensor(0.75)


# class LIF_no_thres(nn.Module):
#     """
#     Leaky-integrate-and-fire neuron with learnable parameters.
#     """

#     def __init__(self, size):
#         super().__init__()
#         self.size = size
#         # Initialize all parameters randomly as U(0, 1)
#         # self.v_decay = nn.Parameter(torch.rand(size))
#         self.v_decay = nn.Parameter(torch.FloatTensor(size).uniform_(0.3, 0.4)) #here
#     def forward(self, synapse, z, state = None):
#         # Previous state
#         if state is None:
#             state = LIFState_no_thresh(
#                 v=torch.zeros_like(synapse(z)),
#             )
#         # Update state
#         i = synapse(z)
#         v = -0.8 + (state.v * self.v_decay + i) * 1.3 
#         return v, LIFState_no_thresh(v)

# class non_SpikingMLP(nn.Module):
#     """
#     Spiking network with LIF neuron model.
#     """

#     def __init__(self, l_1_size, output_size):
#         super().__init__()
#         # self.sizes = sizes
#         # Define layers
#         self.synapses = nn.ModuleList()
#         self.neurons = nn.ModuleList()
#         self.states = []
#         # Loop over current (accessible with 'size') and next (accessible with 'sizes[i]') element
#         # for i, size in enumerate(sizes[-1], start=1):
#             # Parameters of synapses and neurons are randomly initialized
#         print(output_size)
#         self.synapses.append(nn.Linear(l_1_size, output_size, bias=False))
#         self.neurons.append(LIF_no_thres(output_size))
#         self.states.append(None)

#     def forward(self, z):
#         for i, (neuron, synapse) in enumerate(zip(self.neurons, self.synapses)):
#             v, self.states[i]  = neuron(synapse, z, self.states[i])
#         return v

#     def reset(self):
#         """
#         Resetting states when you're done is very important!
#         """
#         for i, _ in enumerate(self.states):
#             self.states[i] = None




# class SpikingMLP(nn.Module):
#     """
#     Spiking network with LIF neuron model.
#     """

#     def __init__(self, sizes):
#         super().__init__()
#         self.sizes = sizes
#         self.spike = SpikeFunction.apply

#         # Define layers
#         self.synapses = nn.ModuleList()
#         self.neurons = nn.ModuleList()
#         self.states = []
#         # Loop over current (accessible with 'size') and next (accessible with 'sizes[i]') element
#         for i, size in enumerate(sizes[:-1], start=1):
#             # Parameters of synapses and neurons are randomly initialized
#             self.synapses.append(nn.Linear(size, sizes[i], bias=False))
#             self.neurons.append(LIF(sizes[i]))
#             self.states.append(None)

#     def forward(self, z):
#         for i, (neuron, synapse) in enumerate(zip(self.neurons, self.synapses)):
#             z, self.states[i] = neuron(synapse, z, self.states[i])
#         return z

#     def reset(self):
#         """
#         Resetting states when you're done is very important!
#         """
#         for i, _ in enumerate(self.states):
#             self.states[i] = None


# class SNN(nn.Module):
#     """
#     Spiking network with LIF neuron model.
#     Has a linear output layer
#     """
#     def __init__(self, snn_sizes, output_size):
#         super().__init__()
#         self.snn_sizes = snn_sizes
#         self.output_size = output_size

#         self.snn = SpikingMLP(snn_sizes)
#         # self.fc = nn.Linear(snn_sizes[-1], output_size)
#         # self.tanh = nn.Tanh()

#         self.last_layer = non_SpikingMLP(snn_sizes[-1], output_size)

#     def forward(self, z):
#         z = self.snn(z)
        
#         # z = self.fc(z)
#         out = self.last_layer(z)
#         return out

#     def set_weights(self, weights):
#         self.weights = weights

#     def reset(self):
#         self.snn.reset()
#         #last layer reset?




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
                print(input_neuron,output_neuron)
                self.edges.append((input_neuron.id, output_neuron.id, 1.))

        # self.networkx_network = nx.from_edgelist(self.edges)
        self.networkx_network = nx.DiGraph()
        # self.networkx_network.add_edges_from(self.edges)
        self.networkx_network.add_weighted_edges_from(self.edges)

        self.fitness = 0.

        self.neuron_matrix = np.full((2,3), 0.03)
# a.species[0].find_all_routes(a.species[0].genomes[0])

        # network stuff
        # self.flatten = nn.Flatten()
    

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
        print(compatibility_score)
        return compatible


    def get_excess_genes(self, comparison_genome):
        excess_genes = []
        largest_innovation_id = max(self.genes.keys())

        for g_id, genome in comparison_genome.genes.items():
            if g_id > largest_innovation_id:
                excess_genes.append(genome)

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


    def mutate(self):
        # Genome Weight Mutations
        for gene in self.genes.values():
            gene.mutate_weight()
        
        for neuron in self.neurons.values():
            neuron.mutate_decay()
            neuron.mutate_threshold()


        # Genome Structural Mutations
        # Adding Gene
        if np.random.uniform() < config.ADD_GENE_MUTATION:

            gene_added = False
            while not gene_added:

                # No valid genes exist to be added.
                if (len(self.hidden_neurons) == 0):
                    break

                # Certain genes are not valid, such as any gene going to an input node, or any gene from an output node
                selected_input_node = np.random.choice(list(set().union(self.hidden_neurons, self.input_neurons)))
                selected_output_node = np.random.choice(list(set().union(self.hidden_neurons, self.output_neurons)))


                neurons_lst_order = list(set().union(self.hidden_neurons, self.output_neurons))
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
                    # this should be changed
                if nx.has_path(self.networkx_network, selected_output_node.id, selected_input_node.id):
                    gene_valid = False

                if gene_valid:
                    new_gene = Gene(self.innovation.get_new_innovation_number(),
                                    selected_input_node,
                                    selected_output_node)

                    self.networkx_network.add_edge(selected_input_node.id, selected_output_node.id, weight=-1.)
                    self.genes[new_gene.innovation_number] = new_gene
                    gene_added = True

        # Adding Neuron
        if np.random.uniform() < config.ADD_NODE_MUTATION:

            # Select gene at random and disable
            selected_gene = np.random.choice(list(self.genes.values()))
            
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
        self.networkx_network.add_weighted_edges_from(self.edges)
    # def find_node_snake(self, next_nodes):





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