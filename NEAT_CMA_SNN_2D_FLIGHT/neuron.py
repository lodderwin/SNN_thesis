import config as config
import numpy as np
import math



class Neuron(object):

    #give position here?
    def __init__(self, neuron_id, x_position=1., y_position=1., z_position=1., n_type="Hidden"):
        self.id = neuron_id
        self.type = n_type

        # Innovation Number : Gene
        self.input_genes = {}
        self.output_genes = {}

        self.received_inputs = 0
        self.input = 0.0
        self.sent_output = False

        #new addition SNN for NEAT
        self.v_decay = np.random.uniform(0.1, 0.9)
        self.threshold = np.random.uniform(0.4, 0.8)

        #new addition for determining place in 3D space 
        # calculate new position that is between two nodes

        self.x_position = x_position
        self.y_position = y_position
        self.z_position = z_position

        # output neuron in the middle of the screen
        # use columns of matrix, and rows to calculate distance

    def expected_inputs(self):
        return 1 if self.type == "Input" else len(self.input_genes)



    # def ready(self):
    #     received_all_inputs = (self.received_inputs == self.expected_inputs())
    #     return (not self.sent_output and received_all_inputs)


    # def fire(self):
    #     self.sent_output = True
    #     for gene in self.output_genes.values():
    #         gene.output_neuron.add_input((self.activation() * gene.weight) if gene.enabled else 0)


    # def has_fired(self):
    #     return self.sent_output


    # def reset_neuron(self):
    #     self.received_inputs = 0
    #     self.input = 0.0
    #     self.sent_output = False


    def add_input(self, value):
        self.input += value
        self.received_inputs += 1


    def add_input_gene(self, gene):
        self.input_genes[gene.innovation_number] = gene


    def add_output_gene(self, gene):
        self.output_genes[gene.innovation_number] = gene


    def mutate_decay(self):
        if np.random.uniform() < config.V_DECAY_MUTATION:
            self.v_decay = np.random.uniform(0.1, 0.9)
        
    def mutate_threshold(self):
        if np.random.uniform() < config.THRESHOLD_MUTATION:
            self.v_decay = np.random.uniform(0.4, 0.8)



    # def activation(self):
    #     return self.sigmoid(self.input)


    # def sigmoid(self, x):
    #     return (2.0 / (1.0 + np.exp(-4.9 * x)) - 1.0)


    def set_id(self, new_id):
        self.id = new_id

