#%%
from email.headerregistry import HeaderRegistry
import torch 
import torch.nn as nn
from collections import namedtuple
from typing import Optional, NamedTuple, Tuple, Any, Sequence
from numpy.core.numeric import outer
# from quad_hover import QuadHover
from quad_hover_var_div import QuadHover
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time

class SpikeFunction(torch.autograd.Function):
    """
    Spiking function with rectangular gradient.
    Source: https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full
    Implementation: https://github.com/combra-lab/pop-spiking-deep-rl/blob/main/popsan_drl/popsan_td3/popsan.py
    """

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)  # save voltage - thresh for backwards pass
        return v.gt(0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (v.abs() < 0.5).float()  # 0.5 is the width of the rectangle
        return grad_input * spike_pseudo_grad, None  # ensure a tuple is returned

#cant call numpy on tensor thatrequires grad. use tensor.detach().numpy()

# class SpikeFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(v):
#         return v.gt(0.0).float()


# # Placeholder for LIF state
LIFState = namedtuple('LIFState', ['z', 'v'])


class LIF(nn.Module):
    """
    Leaky-integrate-and-fire neuron with learnable parameters.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        # Initialize all parameters randomly as U(0, 1)
        # self.v_decay = nn.Parameter(torch.rand(size))
        self.v_decay = nn.Parameter(torch.FloatTensor(size).uniform_(0.3, 0.85)) #here
        # self.thresh = nn.Parameter(torch.rand(size))
        self.thresh = nn.Parameter(torch.FloatTensor(size).uniform_(0.7, 0.999))  #here
        self.spike = SpikeFunction.apply  # spike function

    def forward(self, synapse, z, state = None):
        # Previous state
        if state is None:
            state = LIFState(
                z=torch.zeros_like(synapse(z)),
                v=torch.zeros_like(synapse(z)),
            )
        # Update state
        i = synapse(z)
        v = state.v * self.v_decay * (1.0 - state.z) + i
        z = self.spike(v - self.thresh)
        
        return z, LIFState(z, v)

###########
class LIFState_no_thresh(NamedTuple):
    v: torch.Tensor

# Placeholder for LIF parameters
class LIFParameters_no_tresh(NamedTuple):
    v_decay: torch.Tensor = torch.as_tensor(0.75)


class LIF_no_thres(nn.Module):
    """
    Leaky-integrate-and-fire neuron with learnable parameters.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        # Initialize all parameters randomly as U(0, 1)
        # self.v_decay = nn.Parameter(torch.rand(size))
        self.v_decay = nn.Parameter(torch.FloatTensor(size).uniform_(0.3, 0.4)) #here
    def forward(self, synapse, z, state = None):
        # Previous state
        if state is None:
            state = LIFState_no_thresh(
                v=torch.zeros_like(synapse(z)),
            )
        # Update state
        i = synapse(z)
        v = -0.8 + (state.v * self.v_decay + i) * 1.3 
        return v, LIFState_no_thresh(v)

class non_SpikingMLP(nn.Module):
    """
    Spiking network with LIF neuron model.
    """

    def __init__(self, l_1_size, output_size):
        super().__init__()
        # self.sizes = sizes
        # Define layers
        self.synapses = nn.ModuleList()
        self.neurons = nn.ModuleList()
        self.states = []
        # Loop over current (accessible with 'size') and next (accessible with 'sizes[i]') element
        # for i, size in enumerate(sizes[-1], start=1):
            # Parameters of synapses and neurons are randomly initialized
        print(output_size)
        self.synapses.append(nn.Linear(l_1_size, output_size, bias=False))
        self.neurons.append(LIF_no_thres(output_size))
        self.states.append(None)

    def forward(self, z):
        for i, (neuron, synapse) in enumerate(zip(self.neurons, self.synapses)):
            v, self.states[i]  = neuron(synapse, z, self.states[i])
        return v

    def reset(self):
        """
        Resetting states when you're done is very important!
        """
        for i, _ in enumerate(self.states):
            self.states[i] = None

#######

# class non_SpikingMLP_double(nn.Module):
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
#         for i, size in enumerate([2], start=1):
#             # Parameters of synapses and neurons are randomly initialized
#             self.synapses.append(nn.Linear(size, sizes[i], bias=False))
#             self.neurons.append(LIF_no_thres(sizes[i]))
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


class SpikingMLP(nn.Module):
    """
    Spiking network with LIF neuron model.
    """

    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.spike = SpikeFunction.apply

        # Define layers
        self.synapses = nn.ModuleList()
        self.neurons = nn.ModuleList()
        self.states = []
        # Loop over current (accessible with 'size') and next (accessible with 'sizes[i]') element
        for i, size in enumerate(sizes[:-1], start=1):
            # Parameters of synapses and neurons are randomly initialized
            self.synapses.append(nn.Linear(size, sizes[i], bias=False))
            self.neurons.append(LIF(sizes[i]))
            self.states.append(None)

    def forward(self, z):
        for i, (neuron, synapse) in enumerate(zip(self.neurons, self.synapses)):
            z, self.states[i] = neuron(synapse, z, self.states[i])
        return z

    def reset(self):
        """
        Resetting states when you're done is very important!
        """
        for i, _ in enumerate(self.states):
            self.states[i] = None


class SNN(nn.Module):
    """
    Spiking network with LIF neuron model.
    Has a linear output layer
    """
    def __init__(self, snn_sizes, output_size):
        super().__init__()
        self.snn_sizes = snn_sizes
        self.output_size = output_size

        self.snn = SpikingMLP(snn_sizes)
        # self.fc = nn.Linear(snn_sizes[-1], output_size)
        # self.tanh = nn.Tanh()

        self.last_layer = non_SpikingMLP(snn_sizes[-1], output_size)

    def forward(self, z):
        z = self.snn(z)
        
        # z = self.fc(z)
        out = self.last_layer(z)
        return out

    def set_weights(self, weights):
        self.weights = weights

    def reset(self):
        self.snn.reset()
        #last layer reset?


class CMA_ES:
    def __init__(
        self, 
        function, 
        N,
        xmean
        ): 
        
         #give weights
        self.N = N
        # self.xmean = xmean

        # if xmean==0:
        #     self.xmean = np.random.uniform(0.2, 0.7, size=(1, self.N)).reshape(-1,1)
        # else:
        self.xmean = xmean # hier staat gewicht van synapse


        self.stopfitness = -1.6
        self.stopeval = 1e3*self.N**2

        self.lamba = int(4 + np.floor(3*np.log(self.N))) ####AAAAAANGEPAAAASTTT LET OP TODO: check dit even yo
        self.sigma = 0.5
        self.mu = np.floor(self.lamba/2)
        self.weights = np.log(self.mu+1) - np.log(np.arange(1, self.mu + 1)).reshape(-1,1)
        self.weights = self.weights/np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2/np.sum(self.weights**2)

        self.cc = 4/(N + 4)
        self.cs = (self.mueff + 2)/(self.N + self.mueff + 3)
        self.mucov = self.mueff
        self.ccov = (1/self.mucov) * 2/(self.N + 1.4)**2 + (1- 1/self.mucov) * ((2*self.mueff - 1)/((self.N + 2 )**2 + 2 * self.mueff))
        self.damps = 1 + 2 * np.max([0., np.sqrt((self.mueff - 1)/(self.N+1)) - 1 ]) + self.cs
        self.pc = np.zeros((self.N, 1))
        self.ps = np.zeros((self.N, 1))
        self.B = np.identity(self.N) 
        self.D = np.identity(self.N)
        self.C = np.dot(np.dot(self.B,self.D),np.transpose(np.dot(self.B,self.D)))
        self.eigeneval = 0.
        self.chiN = np.sqrt(self.N) * (1 - 1/(4*self.N)+ 1 / (21*self.N**2))

        self.arz = np.zeros((int(self.N), self.lamba))
        self.arx = np.zeros((int(self.N), self.lamba))
        self.arfitness = np.zeros((self.lamba))
        self.arindex = np.zeros(self.lamba)

        self.array_plot = []

        self.counteval = 0.

        self.function = function

    def optimize_run(self, runs, div):
        for i in range(runs):
            print('gogogo', i)
            for k in range(int(self.lamba)):
                self.arz[:,k] = np.random.normal(0., 0.1, size=(1,self.N)) #0.333  #does not necessarily have to fit column
                # self.arz[:,k] = np.zeros(self.lamba) + 0.2
                self.arx[:,k] = self.xmean.squeeze() + (self.sigma * (np.dot(np.dot(self.B,self.D),self.arz[:,k].reshape(-1,1)))).squeeze()

                ########### constraint
                self.arx[self.arx > 1.] = 0.89999
                self.arx[self.arx < 0.] = 0.0001

                self.arfitness[k] = self.function(self.arx[:,k], div[0]) + self.function(self.arx[:,k], div[1]) + self.function(self.arx[:,k], div[2]) + self.function(self.arx[:,k], div[3]) + self.function(self.arx[:,k], div[4]) 
                self.counteval = self.counteval + 1 

            # Expressing MOO objectives reward easy when considering that the 3D rates have to be a certain level--> more stable evolution

            ##### Potential changes for MOO or 1+ lambda CMA ES
            self.arindex = np.argsort(self.arfitness) #minimazation
            self.arfitness = np.sort(self.arfitness)

            # self.arfitness, self.arindex = self.arfitness[::-1], self.arindex[::-1]

            self.xmean = np.dot(self.arx[:,[self.arindex[:len(self.weights)]]],self.weights).reshape(self.N,1)
            self.zmean = np.dot(self.arz[:,[self.arindex[:len(self.weights)]]],self.weights).reshape(self.N,1)

            self.ps = (1-self.cs)*self.ps + (np.sqrt(self.cs*(2-self.cs)*self.mueff))*np.dot(self.B,self.zmean)
            self.hsig = int(np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.counteval/self.lamba))/self.chiN < 1.5+1/(self.N-0.5))
            self.pc = (1-self.cc)*self.pc + self.hsig * np.sqrt(self.cc*(2.-self.cc)*self.mueff) * np.dot(np.dot(self.B,self.D), self.zmean)

            self.C = (1-self.ccov) * self.C + self.ccov *(1/self.mucov) * \
                    (np.dot(self.pc, np.transpose(self.pc))+ (1 - self.hsig)*self.cc*(2-self.cc) * self.C) \
                        + self.ccov * (1-(1/self.mucov)) * np.dot(np.dot(np.dot(np.dot(self.B, self.D), self.arz[:,[self.arindex[:int(self.mu)]]].squeeze()), np.diag(self.weights.squeeze())), np.transpose(np.dot(np.dot(self.B, self.D), self.arz[:,[self.arindex[:int(self.mu)]]].squeeze() ) ))


            self.sigma = self.sigma * np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chiN - 1))

            if self.counteval - self.eigeneval > (self.lamba/self.ccov/self.N/10.) :
                self.eigeneval = self.counteval
                self.C = np.triu(self.C) + np.transpose(np.triu(self.C, 1))
                self.B, self.D = np.linalg.eig(self.C)[1], np.diag(np.linalg.eig(self.C)[0])  #check later if order of eigenvalues matter??????? I don't think so
                self.D = np.diag(   np.linalg.eig(np.sqrt(np.diag(np.linalg.eig(self.D)[0])))[0])

            # if self.arfitness[0]<=self.stopfitness:
                # break

            if self.arfitness[0] == self.arfitness[int(min(1+ np.floor(self.lamba/2), 2+np.ceil(self.lamba/4.)))]:
                self.sigma = self.sigma * np.exp(0.2+self.cs/self.damps)
                print('gp')
            
            print(self.counteval, self.arfitness[0], 'worst :', self.arfitness[-1])
            self.array_plot.append([self.arx[0,self.arindex[0]], self.arx[1,self.arindex[0]]])

        #write weights to pickle
            weights = self.arx
            best_fitness = self.arfitness[0]

        return weights[:,0], best_fitness



# def model_weights_as_vector(model):  #to initialize xmean
    
#     # select only weights
#     weights_vector = []
#     # print(model.state_dict()['snn.synapses.0.weight'])
#     # print([value for key, value in model.state_dict().items() if 'weight' in key.lower()])
#     for curr_weights in model.state_dict().values():
#         # Calling detach() to remove the computational graph from the layer. 
#         # numpy() is called for converting the tensor into a NumPy array.
#         curr_weights = curr_weights.detach().numpy()
#         vector = np.reshape(curr_weights, newshape=(curr_weights.size))
#         weights_vector.extend(vector)
#     # print(weights_vector)
#     return np.array(weights_vector)

def model_weights_as_vector(model):  #initialise xmean
    
    # select only weights
    weights_vector = []

    to_be_adjusted_weights_only = [key for key, value in model.state_dict().items() if 'snn' in key.lower()]
    for key in to_be_adjusted_weights_only:
        # for curr_weights in model.state_dict()[key]:
            # Calling detach() to remove the computational graph from the layer. 
            # numpy() is called for converting the tensor into a NumPy array.
        curr_weights = model.state_dict()[key].detach().numpy()
        vector = np.reshape(curr_weights, newshape=(curr_weights.size))
        weights_vector.extend(vector)
    return np.array(weights_vector)

def model_weights_as_dict(model, weights_vector): #input

    to_be_adjusted_weights_only = [key for key, value in model.state_dict().items() if 'snn' in key.lower()]
    weights_dict = model.state_dict()
    start = 0
    for key in weights_dict:
        if key in to_be_adjusted_weights_only:
            # Calling detach() to remove the computational graph from the layer. 
            # numpy() is called for converting the tensor into a NumPy array.
            w_matrix = weights_dict[key].detach().cpu().numpy()
            layer_weights_shape = w_matrix.shape
            layer_weights_size = w_matrix.size

            layer_weights_vector = weights_vector[start:start + layer_weights_size]
            layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
            weights_dict[key] = torch.from_numpy(layer_weights_matrix)

            start = start + layer_weights_size
    return weights_dict

# def model_weights_as_vector(model):  #initialise xmean
    
#     # select only weights
#     weights_vector = []

#     to_be_adjusted_weights_only = [key for key, value in model.state_dict().items() if 'weight' in key.lower()]
#     for key in to_be_adjusted_weights_only:
#         # for curr_weights in model.state_dict()[key]:
#             # Calling detach() to remove the computational graph from the layer. 
#             # numpy() is called for converting the tensor into a NumPy array.
#         curr_weights = model.state_dict()[key].detach().numpy()
#         vector = np.reshape(curr_weights, newshape=(curr_weights.size))
#         weights_vector.extend(vector)
#     return np.array(weights_vector)

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

# def model_weights_as_dict(model, weights_vector): #input

#     to_be_adjusted_weights_only = [key for key, value in model.state_dict().items() if 'weight' in key.lower()]
#     weights_dict = model.state_dict()
#     start = 0
#     for key in weights_dict:
#         if key in to_be_adjusted_weights_only:
#             # Calling detach() to remove the computational graph from the layer. 
#             # numpy() is called for converting the tensor into a NumPy array.
#             w_matrix = weights_dict[key].detach().cpu().numpy()
#             layer_weights_shape = w_matrix.shape
#             layer_weights_size = w_matrix.size

#             layer_weights_vector = weights_vector[start:start + layer_weights_size]
#             layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
#             weights_dict[key] = torch.from_numpy(layer_weights_matrix)

#             start = start + layer_weights_size
#     return weights_dict

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

def build_model(param, model):
    with torch.no_grad():
        #alter function to only adjust weights, set biases to 0 
        weight_dict = model_weights_as_dict(model, param)
        model.load_state_dict(weight_dict)
    return model

def build_model_param(param, model, parameter):
    with torch.no_grad():
        # alter function to only adjust weights, set biases to 0
        weight_dict = model_parameter_as_dict(model, param, parameter)
        model.load_state_dict(weight_dict)
    return model


class objective:
    def __init__(self, model, environment, later):
        self.model = model

        self.N_decay = int(len(model_parameter_as_vector(model, 'v_decay')))
        self.N_threshold = int(len(model_parameter_as_vector(model, 'thresh')))

        self.N_weights = int(len(model_parameter_as_vector(model, 'weight')))
        self.N = int(self.N_decay + self.N_threshold + self.N_weights)

        self.model = self.model.to(device=device)

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


    def objective_function(self, x, ref_div):  # add prob here
        # for i in [1.0, 1.5]:
        # ref_div = i
        steps=100000
        
        # prob_ref = np.random.uniform(0.5, 1.5)*0.5
        # mav_model = build_model(x, self.model)

        
        mav_model = self.model
        x_decay, x_thresh, x_weights = x[:self.N_decay], x[self.N_decay-1:self.N_threshold], x[self.N_threshold-1:]
        mav_model = build_model_param(x_decay, mav_model, 'v_decay')
        mav_model = build_model_param(x_thresh, mav_model, 'threshold')
        mav_model = build_model_param(x_weights, mav_model, 'weight')

        # self.environment.ref_div = prob_ref*2
        self.environment.ref_div = ref_div
        prob_ref = self.environment.ref_div/2.

        self.environment.reset()
        divs_lst = []
        ref_lst = []
        # print(mav_model.state_dict())

        reward_cum = 0
        for step in range(steps):
            encoded_input = self.position_encoder(list(self.environment.obs)[-2:], prob_ref)


            # print(encoded_input)
            thrust_setpoint = mav_model(encoded_input.float()) 
            # print(thrust_setpoint)           
            divs, reward, done, _, _ = self.environment.step(thrust_setpoint.detach().numpy())
            self.environment._get_reward()    
            self.environment.render()
            if done:
                break
            divs_lst.append(self.environment.state[0])
            ref_lst.append(self.environment.height_reward)
            # divs_lst.append(self.environment.thrust_tc)
        
        mav_model.reset()    

        # time.sleep(0.1)
        plt.plot(divs_lst, c='#4287f5')
        plt.plot(ref_lst, c='#f29b29')

        
        plt.ylabel('height (m)')
        plt.xlabel('timesteps (0.02s)')
        plt.title('Divergence: '+ str(ref_div))
        # plt.plot(divs_lst)
        
        # figure(figsize=(8, 6), dpi=80)
        # plt.show()
        plt.savefig('pres_1.png')
        # plt.show()
        reward_cum = self.environment.reward
        print(reward_cum, self.environment.state[0], )
        return reward_cum



# class organize_training():
#     def __init__(self, objective, init_training):

#         # self.optimizer = optimizer
#         self.objective = objective
#         self.objective_function = self.objective.objective_function
#         # give 'landing speed'

#         self.initialisation_values = init_training

#     def write_weights(self, weights):
#         np.savetxt('new_try.txt', weights)
    
#     def read_weights(self, filename):
#         weights = np.load(filename)
#         return weights

#     def initialise_from_scratch(self):
#         x_decay = np.random.normal(self.initialisation_values[0], 0.1, size=(1, self.objective.N_decay)).squeeze()
#         # self.objective.model = build_model_param(x_decay, self.objective.model, 'v_decay')
#         x_thresh = np.random.normal(self.initialisation_values[1], 0.2, size=(1, self.objective.N_threshold)).squeeze()
#         # self.objective.model = build_model_param(x_thresh, self.objective.model, 'thresh')
#         x_weights = np.random.normal(self.initialisation_values[2], 0.2, size=(1, self.objective.N_weights)).squeeze()
#         # self.objective.model = build_model_param(x_weights, self.objective.model, 'weight')

#         xmean = np.concatenate((x_decay, x_thresh, x_weights), axis=0)
#         print(xmean)

#         self.optimizer = CMA_ES(self.objective_function, self.objective.N, xmean)
#         weights = self.optimizer.optimize_run(50, 1.)
#         self.write_weights(weights)
    
#     # def train_multiple_divs(self):
class organize_training():
    def __init__(self, objective, init_training):

        # self.optimizer = optimizer
        self.objective = objective
        self.objective_function = self.objective.objective_function
        # give 'landing speed'

        self.initialisation_values = init_training

        self.worst_performance = 1000.
        self.best_init_weights = []

    def write_weights(self, weights):
        np.savetxt('new_try.txt', weights)
    
    def read_weights(self, filename):
        weights = np.load(filename)
        return weights

    def initialise_from_scratch(self):
        for i in range(10):
            x_decay = np.random.uniform(self.initialisation_values[0]-0.1, self.initialisation_values[0] + 0.1, size=(1, self.objective.N_decay)).squeeze()
            # self.objective.model = build_model_param(x_decay, self.objective.model, 'v_decay')
            x_thresh = np.random.uniform(self.initialisation_values[1]-0.1, self.initialisation_values[1] + 0.1, size=(1, self.objective.N_threshold)).squeeze()
            # self.objective.model = build_model_param(x_thresh, self.objective.model, 'thresh')
            x_weights = np.random.uniform(self.initialisation_values[2]-0.1, self.initialisation_values[2] + 0.1, size=(1, self.objective.N_weights)).squeeze()
            # self.objective.model = build_model_param(x_weights, self.objective.model, 'weight')

            xmean = np.concatenate((x_decay, x_thresh, x_weights), axis=0)
            self.optimizer = CMA_ES(self.objective_function, self.objective.N, xmean)
            weights, best_fitness = self.optimizer.optimize_run(1, 1.)
            if best_fitness<self.worst_performance:
                self.write_weights(weights)
                self.best_init_weights = weights
        return self.best_init_weights

        

    def multiple_div(self, xmean):
        self.optimizer = CMA_ES(self.objective_function, self.objective.N, xmean)
        for i in range(10):
            div_training = [np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5)]

            weights, best_fitness = self.optimizer.optimize_run(25, div_training)
        return weights


    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running network on the following device: {device}')

model = SNN([5, 10, 5], 1)

# build_model() # every time
environment = QuadHover()


# model.state_dict().values()
        
# initialise weight of vdecay threshold here so they remain unchanged
objective = objective(model, environment, 'weight')
objective_function = objective.objective_function

weights_1 = np.load('pres_old_config.txt.npy')

# weights_1 = np.random.uniform(0.3, 0.8, size=(1, objective.N)).reshape(-1,1)
# write function feed all pro
# optimizer = CMA_ES(objective_function, objective.N, xmean=weights_1.reshape(-1,1))
# weights = optimizer.optimize_run(runs=100)

# np.save('go.txt', weights)





init_training = [0.15840909, 0.755914, 0.45380776]
# init_training = [0.12130266, 0.61771752, 0.30521799]
# init_training = [0.75840909, 0.155914, 0.15380776]
# init_training = [0.12130266, 0.61771752, 0.30521799] #model = SNN([5, 10, 5], 1)

organize_training = organize_training(objective, init_training)
# best_weights = organize_training.initialise_from_scratch()


# plt.show()
best_weights_final = organize_training.multiple_div(weights_1.reshape(-1,1))

# best score for mutliple divergences should be taken into account for CMA-ES

# np.save('pres_old_config.txt', best_weights_final)
# organize_training.objective_function(weights_1.reshape(-1,1), 1.0)
# plot landings --> plot random landings to compare performance. Reference landing and performed landing
# new SNN network with new last layer mechanism todo next time
# trainer: sim for the same div in CMA-ES otherwise you can not completely compare performances


# It's all about finding tipping point in intialized values, easy to vconvert from that point onwards.
# %%


#design function for weight
# fit normal distribution to divergence, 0.5, 1.5 normal, reward dependend on divergence.
# s = np.random.normal(mu, sigma, 1000)
# 1 0.5, 