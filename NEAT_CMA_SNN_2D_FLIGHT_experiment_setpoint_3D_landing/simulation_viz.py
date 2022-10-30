from network_viz import draw_net
from species import find_all_routes, clean_array, place_weights
from env_3d_var_D_setpoint import LandingEnv3D as Quadhover
from snn_pytorch_mod_double_output import SNN
import torch
import numpy as np
import dill
import matplotlib.pyplot as plt
from visualize import DrawNN

from species import objective as objective_cma
from CMAES_NEAT import CMA_ES, CMA_ES_single


class objective:
    def __init__(self, model, environment):
        self.model = model
        self.environment = environment

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

    def objective_function_single(self, model, ref_div, ref_wx, ref_wy):  # add prob here

        steps=100000
        
        mav_model = model
        # self.environment.ref_div = prob_ref*2
        self.environment.ref_div = ref_div
        self.environment.ref_wx = ref_wx
        self.environment.ref_wy = ref_wy


        self.environment.reset()
        ref_vertical = []
        ref_horizontal = []
        act_vertical = []
        act_horizontal = []
        thrust_setpoint = []
        reward_cum = 0
        for step in range(steps):
            prob_x = self.environment.x_prob()
            prob_z = self.environment.z_prob()
            prob_y = self.environment.y_prob()
            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_x, prob_z, prob_y)

            array = mav_model(encoded_input.float()) 
            control_input = self.spike_decoder(array.detach().numpy())
            # print(control_input) #control_input[1]
            divs, reward, done, _, _ = self.environment.step(np.asarray([0., control_input[1], control_input[1]]))
            # divs, reward, done, _, _ = self.environment.step(np.asarray([0., 0., 0.]))
            # self.environment._get_reward()    
            # self.environment.render()
            print(reward)
            if done:
                break
            ref_horizontal.append(self.environment.forward_reward)
            ref_vertical.append(self.environment.height_reward)
            act_horizontal.append(self.environment.state[0][0])
            act_vertical.append(self.environment.state[2][0])

            thrust_setpoint.append(control_input[1])
            # divs_lst.append(self.environment.thrust_tc)
        
        plt.plot(ref_horizontal,ref_vertical, c='#93a6f5')
        plt.plot(act_horizontal,act_vertical,  c='#2b54ff')
        # plt.plot(thrust_setpoint)
        
        plt.ylabel('height (m)')
        plt.xlabel('distance (m)')
        plt.title('Divergence: '+ str(ref_div))
        # plt.savefig('25-08meeting.png')
        mav_model.reset()    
        reward_cum = self.environment.reward
        print(reward_cum, self.environment.state[3][0], self.environment.state[5][0] )
        return reward_cum

    def objective_function_multiple(self, model, ref_div, ref_wx, ref_wy, runs):  # add prob here
        all_runs_vertical = []
        all_runs_horizontal = []

        nearest_traj = 100.
        nearest_traj_int = 0
        furthest_traj = 0.
        furthest_traj_int = 0
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca(projection='3d')
        for run in range(runs):

            steps=100000
            
            mav_model = model
            # self.environment.ref_div = prob_ref*2
            self.environment.ref_div = ref_div
            self.environment.ref_wx = ref_wx
            self.environment.ref_wy = ref_wy
            prob_ref_div = self.environment.ref_div/2.*2.
            prob_ref_wx = self.environment.ref_wx/2.*2.

            self.environment.reset()
            ref_vertical = []
            ref_horizontal = []
            act_vertical = []
            act_horizontal = []
            ref_horizontal_uncoupled = []
            act_side = []

            input_control = []
            reward_cum = 0
            for step in range(steps):
                prob_x = self.environment.x_prob()
                prob_z = self.environment.z_prob()
                prob_y = self.environment.y_prob()
                encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_x, prob_z, prob_y)

                array = mav_model(encoded_input.float()) 
                control_input = self.spike_decoder(array.detach().numpy())
                divs, reward, done, _, _ = self.environment.step(np.asarray([control_input[2], control_input[0], control_input[1]]))
                # divs, reward, done, _, _ = self.environment.step(np.asarray([0., 0., 0.]))
                # self.environment._get_reward()    
                # self.environment.render()
                if done:
                    break
                ref_horizontal.append(self.environment.forward_reward)
                ref_vertical.append(self.environment.height_reward)
                act_horizontal.append(self.environment.state[0][0])
                act_vertical.append(self.environment.state[2][0])
                act_side.append(self.environment.state[1][0])
                # input_control.append(control_input[0])
                # print(self.environment.state[2][0])


                # print(self.environment.state[3][0],control_input[1])
                # ref_horizontal_uncoupled.append(self.environment.forward_reward_adm)
                
                # plt.plot(ref_horizontal_uncoupled,ref_vertical, c='#e89725')
                # divs_lst.append(self.environment.thrust_tc)
            all_runs_vertical.append(act_vertical)
            all_runs_horizontal.append(act_horizontal)

            ax.plot(act_horizontal,act_side,act_vertical, c='#93a6f5', alpha=0.9)
            # print(act_horizontal)




        print(ref_horizontal[-1], ref_vertical[-1])
        print(act_horizontal[-1], act_vertical[-1])
        ax.set_zlabel('height (m)')
        ax.set_xlabel('distance x (m)')
        ax.set_ylabel('distance y (m)')
        plt.title('Divergence: '+ str(ref_div))
        ax.view_init(15, 45)  #(15,90)
        ax.legend()
        plt.savefig('2510meeting.png')
        # plt.plot(input_control)
        print(self.environment.state[3],self.environment.state[5])
        mav_model.reset()    
        reward_cum = self.environment.reward
        print('reward:', reward_cum, self.environment.t)
        return reward_cum

with open('3Dlanding_2_reward_01D.pkl', 'rb') as f:
    neat_class = dill.load(f)

# neat_class.species[neat_class.best_species].genomes[neat_class.best_genome]
neuron_matrix = find_all_routes(neat_class.best_genome)
neuron_matrix = clean_array(neuron_matrix)
model = place_weights(neuron_matrix, neat_class.best_genome)
network_viz = draw_net(neat_class.best_genome)
environment = Quadhover()
objective = objective(model, environment=environment)
objective.objective_function_multiple(model, 0.1, 0.1, 0.1, 5)
# objective.objective_function_single(model, 0.3, 0.3)
print(model.state_dict())
print(neat_class.best_genome.fitness, neat_class.best_genome)


network_viz.view()
# print(neat_class.species[neat_class.best_species].genomes[neat_class.best_genome].fitness, neat_class.best_genome)

environment = Quadhover()
objective_genome = objective_cma(environment)
# # CMA-ES learning 
cycles = 5
tags = list({x[0]: x[1].weight for x in neat_class.best_genome.genes.items()}.keys())
weights = np.asarray(list({x[0]: x[1].weight for x in neat_class.best_genome.genes.items()}.values()))

# print('aii', weights)

cma_es_class  = CMA_ES(objective_genome.objective_function_CMAES, N=weights.shape[0], xmean=weights, genome=neat_class.best_genome)
new_weights, best_fitness = cma_es_class.optimize_run(cycles, [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], neat_class.best_genome)

# print('aai', new_weights)

gene_ad = 0
for gene in tags:
    neat_class.best_genome.genes[gene].weight = new_weights[gene_ad]
    gene_ad = gene_ad + 1
# print(neat_class.best_genome.fitness)


neuron_matrix = find_all_routes(neat_class.best_genome)
neuron_matrix = clean_array(neuron_matrix) 
model = place_weights(neuron_matrix, neat_class.best_genome)
network_viz = draw_net(neat_class.best_genome)
environment = Quadhover()
# objective = objective(model, environment=environment)
objective.objective_function_multiple(model, 0.1, 0.1, 0.1, 1)

# with open('testing_decreasedCMAES_3D_new_control_sys_faster_random_30_4_really_goodone_withCMA_test.pkl', 'wb') as outp:
#     dill.dump(neat_class, outp)


# draw_nn = DrawNN(model)
# draw_nn.draw()

# species = 0
# genome = 9


# neuron_matrix = find_all_routes(neat_class.species[species].genomes[genome])
# neuron_matrix = clean_array(neuron_matrix)
# model = place_weights(neuron_matrix, neat_class.species[species].genomes[genome])
# network_viz = draw_net(neat_class.species[species].genomes[genome])
# environment = Quadhover()
# objective = objective(model, environment=environment)
# objective.objective_function_multiple(model, 0.3, 0.3, 1)
# network_viz.view()
# print(neat_class.species[species].genomes[genome].fitness, neat_class.best_genome)

# draw_nn = DrawNN(model)
# draw_nn.draw()
