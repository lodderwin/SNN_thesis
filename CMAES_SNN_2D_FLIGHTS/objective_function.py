from tkinter import W

from scipy.stats import expon
from env_3d_var import LandingEnv3D as Quadhover
import matplotlib.pyplot as plt
# import FlapPyBird.flappy as flpy
import os
# os.chdir(os.getcwd() + '/FlapPyBird/')
from collections import Counter
from snn_pytorch_mod_double_output import SNN
import torch
import networkx as nx
import copy 
from uuid import uuid4
from CMAES import CMA_ES
import dill

import numpy as np
# uuid4()
# UUID('4a59bf4e-1653-450b-bc7e-b862d6346daa')

class objective:
    def __init__(self, model, environment):
        self.model = model
        # self.model = self.model.to(device=device)
        # self.N_weights = int(len(model_parameter_as_vector(model, 'weight')))

        self.N_decay = int(len(model_parameter_as_vector(model, 'v_decay')))
        self.N_threshold = int(len(model_parameter_as_vector(model, 'thresh')))

        self.N_weights = int(len(model_parameter_as_vector(model, 'weight')))
        self.N = int(self.N_decay + self.N_threshold + self.N_weights)



        self.environment = environment
        


    def spike_encoder_div(self, OF_lst, prob_ref_div, prob_ref_wx):
        #current encoder

        D_plus = bool(OF_lst[-1][2]>0.) * 1.
        D_min = bool(OF_lst[-1][2]<0.) * 1.

        D_delta_plus = bool(OF_lst[-1][2]>OF_lst[-2][2]) * 1.
        D_delta_min = bool(OF_lst[-1][2]<OF_lst[-2][2]) * 1.
####        
        ref_div_node = bool(np.random.uniform()>=(1-prob_ref_div))
####

        wx_plus = bool(OF_lst[-1][0]>0.) * 1.
        wx_min = bool(OF_lst[-1][0]<0.) * 1.

        wx_delta_plus = bool(OF_lst[-1][0]>OF_lst[-2][0]) * 1.
        wx_delta_min = bool(OF_lst[-1][0]<OF_lst[-2][0]) * 1.
####        
        ref_wx_node = bool(np.random.uniform()>=(1-prob_ref_wx))
####

        x = torch.tensor([D_plus, D_min, D_delta_plus, D_delta_min, ref_div_node, wx_plus, wx_min, wx_delta_plus, wx_delta_min, ref_wx_node])

        return x
    def spike_decoder(self, spike_array):


        thrust = spike_array[0] - spike_array[1]
        pitch = spike_array[2] - spike_array[3]

        return (thrust, pitch)
    
    
    def objective_function_CMAES(self, x, ref_div, ref_wx):  # add prob here
        steps=100000
        
        # print(mav_model.state_dict())
        # mav_model = build_model(x, self.model)

        mav_model = self.model
        self.environment.ref_div = ref_div
        self.environment.ref_wx = ref_wx
        prob_ref_div = self.environment.ref_div/2.
        prob_ref_wx = self.environment.ref_wx/2.


        # mav_model = build_model_param(x, mav_model, 'weight')

        x_decay, x_thresh, x_weights = x[:self.N_decay], x[self.N_decay-1:self.N_threshold], x[self.N_threshold-1:]
        mav_model = build_model_param(x_decay, mav_model, 'v_decay')
        mav_model = build_model_param(x_thresh, mav_model, 'threshold')
        mav_model = build_model_param(x_weights, mav_model, 'weight')

        self.environment.reset()
        divs_lst = []
        ref_lst = []
        # print(mav_model.state_dict())

        reward_cum = 0
        for step in range(steps):
            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_ref_div, prob_ref_wx)
            array = mav_model(encoded_input.float()) 
            # print('b',array)
            control_input = self.spike_decoder(array.detach().numpy())
            # print('a', encoded_input, control_input)           
            divs, reward, done, _, _ = self.environment.step(np.asarray([0., control_input[1], control_input[0]]))

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


class organize_training():
    def __init__(self, objective):
        self.objective = objective
        self.objective_function = self.objective.objective_function_CMAES
        # self.x_weights = np.random.uniform(0.3, 0.7, size=(1, self.objective.N_weights)).squeeze()
        # self.optimizer = CMA_ES(self.objective_function, self.objective.N, self.x_weights)
        self.div_training = [np.random.uniform(0.75, 2.0), np.random.uniform(0.75, 2.0), np.random.uniform(0.75, 2.0), np.random.uniform(0.75, 2.0), np.random.uniform(0.75, 2.0)]
        self.wx_training = [np.random.uniform(0.75, 2.0), np.random.uniform(0.75, 2.0), np.random.uniform(0.75, 2.0), np.random.uniform(0.75, 2.0), np.random.uniform(0.75, 2.0)]

        self.initialisation_values = [0.5, 0.7, 0.5]

        self.worst_performance = 1000000000000.
        self.best_ratio = 0.
        self.xmean = []

        self.plot_array = []

    def initialise_from_scratch(self):
        for i in range(100):
            x_decay = np.random.uniform(self.initialisation_values[0]-0.1, self.initialisation_values[0] + 0.3, size=(1, self.objective.N_decay)).squeeze()
            # self.objective.model = build_model_param(x_decay, self.objective.model, 'v_decay')
            x_thresh = np.random.uniform(self.initialisation_values[1]-0.1, self.initialisation_values[1] + 0.1, size=(1, self.objective.N_threshold)).squeeze()
            # self.objective.model = build_model_param(x_thresh, self.objective.model, 'thresh')
            x_weights = np.random.uniform(self.initialisation_values[2]-0.3, self.initialisation_values[2] + 0.3, size=(1, self.objective.N_weights)).squeeze()
            # self.objective.model = build_model_param(x_weights, self.objective.model, 'weight')

            xmean = np.concatenate((x_decay, x_thresh, x_weights), axis=0)
            div_training = self.div_training
            wx_training = self.wx_training
            self.optimizer = CMA_ES(self.objective_function, self.objective.N, xmean)
            weights, best_fitness, worst_fitness = self.optimizer.optimize_run(1, div_training, wx_training)
            print(((worst_fitness-best_fitness)/best_fitness))
            if ((worst_fitness-best_fitness)/best_fitness)>self.best_ratio:
                # self.write_weights(weights)
                self.xmean = weights
                self.best_ratio = ((worst_fitness-best_fitness)/best_fitness)
        return self.xmean

    def init_optimizer(self):
        self.optimizer = CMA_ES(self.objective_function, self.objective.N, np.asarray(self.xmean))
        self.plot_array.append(self.worst_performance)

    def train(self, cycles):
        for i in range(cycles):
            div_training = self.div_training
            wx_training = self.wx_training
            self.xmean, best_fitness, worst_fitness = self.optimizer.optimize_run(10, div_training, wx_training)
            self.plot_array.append(best_fitness)

# with open('CMA_ES_0709.pkl', 'rb') as f:
#     d = dill.load(f)
# model = SNN([10,10], 4)
# environment = Quadhover()
# objective = objective(model, environment)
# organize_training = organize_training(objective=objective)
# organize_training.initialise_from_scratch()
# organize_training.init_optimizer()
# organize_training.train(1)
# with open('CMA_ES_0709.pkl', 'wb') as outp:
#     dill.dump(organize_training, outp)


import time
with open('CMA_ES_0709.pkl', 'rb') as f:
    organize_training = dill.load(f)
for i in range(50):
    start_time = time.time()
    organize_training.train(1)
    with open('CMA_ES_0709.pkl', 'wb') as outp:
        dill.dump(organize_training, outp)
    print("--- %s seconds ---" % (time.time() - start_time))
