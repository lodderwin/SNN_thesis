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
        # plt.savefig('pres_1.png')
        # plt.show()
        reward_cum = self.environment.reward
        print(reward_cum, self.environment.state[0], )
        return reward_cum



#%%
import torch
import torch.nn.functional as F
from torch import nn
import collections
from typing import DefaultDict, Tuple, List, Dict
from functools import partial

def save_activations(
        activations: DefaultDict,
        name: str,
        module: nn.Module,
        inp: Tuple,
        out: torch.Tensor
) -> None:
    """PyTorch Forward hook to save outputs at each forward
    pass. Mutates specified dict objects with each fwd pass.
    """
    # activations[name].append(out.detach().cpu())
    activations[name].append(out[0].detach().cpu())
def register_activation_hooks(
        model: nn.Module,
        layers_to_save: List[str]
) -> DefaultDict[List, torch.Tensor]:
    """Registers forward hooks in specified layers.
    Parameters
    ----------
    model:
        PyTorch model
    layers_to_save:
        Module names within ``model`` whose activations we want to save.

    Returns
    -------
    activations_dict:
        dict of lists containing activations of specified layers in
        ``layers_to_save``.
    """
    activations_dict = collections.defaultdict(list)

    for name, module in model.named_modules():
        if name in layers_to_save:
            module.register_forward_hook(
                partial(save_activations, activations_dict, name)
            )
    return activations_dict

    # for name, module in model._modules['snn'] named_modules()



# class Net(nn.Module):
#     """Simple two layer conv net"""
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 8, kernel_size=(5, 5), stride=(2,2))
#         self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(2,2))

#     def forward(self, x):
#         y = F.relu(self.conv1(x))
#         z = F.relu(self.conv2(y))
#         return z

# mdl = Net()
# to_save = ["conv1", "conv2"]
# to_save = ['snn.neurons.0', 'snn.neurons.1']

# register fwd hooks in specified layers
# saved_activations = register_activation_hooks(model, layers_to_save=to_save)

# run twice, then assert each created lists for conv1 and conv2, each with length 2
# num_fwd = 2
# images = [torch.randn(10, 3, 256, 256) for _ in range(num_fwd)]
# for _ in range(num_fwd):
#     mdl(images[_])

# assert len(saved_activations["conv1"]) == num_fwd
# assert len(saved_activations["conv2"]) == num_fwd

#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running network on the following device: {device}')

model = SNN([5, 3], 1)

to_save = ['snn.neurons.0', 'snn.neurons.1']

# register fwd hooks in specified layers
saved_activations = register_activation_hooks(model, layers_to_save=to_save)

# build_model() # every time
environment = QuadHover()


# model.state_dict().values()
        
# initialise weight of vdecay threshold here so they remain unchanged
objective = objective(model, environment, 'weight')
objective_function = objective.objective_function

# weights_1 = np.load('pres_old_config.txt.npy')

weights_1 = np.random.uniform(0.3, 0.8, size=(1, objective.N)).reshape(-1,1)

objective_function(weights_1, 1.)

# len(save_activations['snn.neurons.0']) is equal to simulation timesteps 



# clean up everything ensure that everything works, including visualization, NEAT EA for 1D environment

# add gene probability connection
# delete gene if weight is very low multiple times
# 