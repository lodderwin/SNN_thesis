import config as config
import numpy as np



class Gene(object):


    def __init__(self, innovation_number, input_neuron, output_neuron, weight=None, enabled=True):

        self.innovation_number = innovation_number

        self.input_neuron = input_neuron
        input_neuron.add_output_gene(self)

        self.output_neuron = output_neuron
        output_neuron.add_input_gene(self)

        self.enabled = enabled
        self.weight = weight

        if self.weight is None:
            self.randomize_weight()


    def mutate_weight(self):
        # Weight Mutation
        if np.random.uniform() < config.WEIGHT_MUTATION_RATE:
            if np.random.uniform() < config.UNIFORM_WEIGHT_MUTATION_RATE:
                self.weight += np.random.uniform(-0.05, 0.05)
            else:
                self.randomize_weight()
                # None

        # Enabled Mutation
        # if not self.enabled:
        #     if np.random.uniform() < config.ENABLE_GENE_MUTATION_RATE:
        #         self.enabled = True

        if self.weight<0.05:
            self.weight = 0.05
        if self.weight>1.0:
            self.weight = 1.0
            
    # pay attention here
    def randomize_weight(self):
        # self.weight = np.random.uniform(-2, 2)
        self.weight = np.random.uniform(0.1, 0.9)

    # def mutate_disable(self):
    #     # genes = [[x.input_neuron.id, x.output_neuron.id] if x.enabled==True else None for x in genome.genes.values()]
    #     # genes = [x for x in genes if x is not None]
    #     output_neuron_input_genes_len = [x if x.enabled==True else None for x in self.output_neuron.input_genes.values()]
    #     output_neuron_input_genes_len = len([x for x in output_neuron_input_genes_len if x is not None])
    #     input_neuron_output_genes_len = [x if x.enabled==True else None for x in self.input_neuron.output_genes.values()]
    #     input_neuron_output_genes_len = len([x for x in input_neuron_output_genes_len if x is not None])
    #     if output_neuron_input_genes_len>=2 and input_neuron_output_genes_len>=2 and np.random.uniform() < config.GENE_REMOVE_RATE:
    #         self.enabled = False

    #     print('GENE DISABLED')

    def disable(self):
        self.enabled = False


    def copy(self):
        return Gene(innovation_number=self.innovation_number,
                    input_neuron_id=self.input_neuron_id,
                    output_neuron_id=self.output_neuron_id,
                    weight=self.weight,
                    enabled=self.enabled)

