from network_viz import draw_net
from species import find_all_routes, clean_array, place_weights
from env_3d_var import LandingEnv3D as Quadhover
from snn_pytorch_mod_double_output import SNN
import torch
import numpy as np
import dill
import matplotlib.pyplot as plt
from visualize import DrawNN



class objective:
    def __init__(self, model, environment):
        self.model = model
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

    def objective_function_NEAT(self, model, ref_div, ref_wx):  # add prob here

        steps=100000
        
        mav_model = model
        # self.environment.ref_div = prob_ref*2
        self.environment.ref_div = ref_div
        self.environment.ref_wx = ref_wx
        prob_ref_div = self.environment.ref_div/2.
        prob_ref_wx = self.environment.ref_wx/2.

        self.environment.reset()
        divs_lst = []
        ref_lst = []

        reward_cum = 0
        for step in range(steps):
            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_ref_div, prob_ref_wx)


            
            array = mav_model(encoded_input.float()) 
            control_input = self.spike_decoder(array.detach().numpy())
            print(control_input)
            # divs, reward, done, _, _ = self.environment.step(np.asarray([0., control_input[1], control_input[0]]))
            divs, reward, done, _, _ = self.environment.step(np.asarray([0., 0., 0.]))
            # self.environment._get_reward()    
            # self.environment.render()
            if done:
                break
            divs_lst.append(self.environment.forward_reward)
            ref_lst.append(self.environment.height_reward)
            # divs_lst.append(self.environment.thrust_tc)
        plt.plot(divs_lst[:200], c='#4287f5')
        plt.plot(ref_lst[:200], c='#f29b29')

        
        plt.ylabel('height (m)')
        plt.xlabel('timesteps (0.02s)')
        plt.title('Divergence: '+ str(ref_div))
        mav_model.reset()    
        reward_cum = self.environment.reward
        print(reward_cum, self.environment.state[0], )
        return reward_cum

with open('demonstration.pkl', 'rb') as f:
    neat_class = dill.load(f)

neuron_matrix = find_all_routes(neat_class.species[neat_class.best_species].genomes[neat_class.best_genome])
neuron_matrix = clean_array(neuron_matrix)
model = place_weights(neuron_matrix, neat_class.species[neat_class.best_species].genomes[neat_class.best_genome])
network_viz = draw_net(neat_class.species[neat_class.best_species].genomes[neat_class.best_genome])
environment = Quadhover()
objective = objective(model, environment=environment)
objective.objective_function_NEAT(model, 1., 1.)
# network_viz.view()


# draw_nn = DrawNN(model)
# draw_nn.draw()
