import torch
import torch.nn as nn
from typing import Optional, NamedTuple, Tuple, Any, Sequence
import numpy as np
from collections import namedtuple



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
        self.v_decay = nn.Parameter(torch.FloatTensor(size).uniform_(0.1, 0.7)) #here
        # self.thresh = nn.Parameter(torch.rand(size))
        self.thresh = nn.Parameter(torch.FloatTensor(size).uniform_(0.7, 0.95))  #here
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
    v_decay: torch.Tensor #= torch.as_tensor(0.75)


class LIF_no_thres(nn.Module):
    """
    Leaky-integrate-and-fire neuron with learnable parameters.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        # Initialize all parameters randomly as U(0, 1)
        # self.v_decay = nn.Parameter(torch.rand(size))
        self.v_decay = nn.Parameter(torch.FloatTensor(size).uniform_(0.8, 0.85)) #here
    def forward(self, synapse, z, state = None):
        # Previous state
        if state is None:
            state = LIFState_no_thresh(
                v=torch.zeros_like(synapse(z)),
            )
        # Update state
        i = synapse(z)
        v = (state.v * self.v_decay + i) 
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
        self.synapses.append(nn.Linear(l_1_size, output_size, bias=False))
        self.neurons.append(LIF_no_thres(output_size))
        self.states.append(None)


        # for i, size in enumerate(sizes[:-1], start=1):
        #     # Parameters of synapses and neurons are randomly initialized
        #     self.synapses.append(nn.Linear(l_1_size, output_size[i], bias=False))
        #     self.neurons.append(LIF_no_thres(output_size))
        #     self.states.append(None)

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