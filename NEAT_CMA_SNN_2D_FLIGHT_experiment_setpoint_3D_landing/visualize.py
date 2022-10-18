from matplotlib import pyplot
from math import cos, sin, atan


class Neuron():
    def __init__(self, x, y, neuron_id):
        self.x = x
        self.y = y

        self.id = neuron_id

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)




class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, weight_array):
        self.horizontal_distance_between_layers = 6
        self.vertical_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()

        self.neuron_id_count = 0
        self.neurons = self.__intialise_neurons(number_of_neurons)

        self.weight_array = weight_array

        

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        y = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(self.x, y, self.neuron_id_count)
            neurons.append(neuron)
            y += self.vertical_distance_between_neurons

            self.new_neuron_id_count()
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.vertical_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_x_position(self):
        if self.previous_layer:
            return self.previous_layer.x + self.horizontal_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def new_neuron_id_count(self):
        self.neuron_id_count += 1

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        print(neuron1.x, neuron1.y, neuron2.x, neuron2.y, linewidth)
        # angle = atan(float(neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        angle = atan(float(neuron2.y - neuron1.y) / float(neuron2.x - neuron1.x))
        x_adjustment = self.neuron_radius * cos(angle)
        y_adjustment = self.neuron_radius * sin(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                
                for previous_layer_neuron in self.previous_layer.neurons:
                    print(neuron.id, previous_layer_neuron.id)
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, self.weight_array[neuron.id, previous_layer_neuron.id])

        # write Text
        # x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        # if layerType == 0:
        #     pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        # elif layerType == -1:
        #     pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        # else:
        #     pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, weight_array ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, weight_array)
        self.layers.append(layer)
        # self.layers.append((layer, ))

    def draw(self):
        pyplot.figure(figsize=(20., 20.))
        
        # pyplot.gca().set_facecolor('white')
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            print('here')
            print(layer.weight_array)
            layer.draw( i )

        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Sparse SNN in MLP form', fontsize=15)
        # pyplot.savefig('networkMLP.png')
        pyplot.show()

class DrawNN():
    def __init__( self, model,):
        self.weight_array = [value.numpy() for key, value in model.state_dict().items() if 'weight' in key.lower()]

        # self.neural_network = neural_network
        self.neural_network = [self.weight_array[0].shape[1]]

        for array in self.weight_array:
            self.neural_network.append(array.shape[0])
        widest_layer = max( self.neural_network )
        self.network = NeuralNetwork( widest_layer )
    def draw( self ):
        for i in range(len(self.neural_network)):
            if i==0:
                self.network.add_layer(self.neural_network[i], None)
            else:
                self.network.add_layer(self.neural_network[i], self.weight_array[i-1])
        self.network.draw()

# draw_nn = DrawNN([4,5])
# draw_nn.draw()