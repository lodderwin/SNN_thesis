from network_viz import draw_net
from species import find_all_routes, clean_array, place_weights
from env_3d_var_point import LandingEnv3D as Quadhover
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


    def spike_encoder_div(self, OF_lst, prob_ref_div, prob_ref_wx, prob_ref_wy):
        #current encoder
        ref_div_node_min = bool(np.random.uniform()>=(1-prob_ref_div[1]))
        D_plus = bool(OF_lst[-1][2]>0.) * 1.
        D_min = bool(OF_lst[-1][2]<0.) * 1.

        D_delta_plus = bool(OF_lst[-1][2]>OF_lst[-2][2]) * 1.
        D_delta_min = bool(OF_lst[-1][2]<OF_lst[-2][2]) * 1.
####        
        ref_div_node_plus = bool(np.random.uniform()>=(1-prob_ref_div[0]))
####

        ref_wx_node_min = bool(np.random.uniform()>=(1-prob_ref_wx[1]))
        wx_plus = bool(OF_lst[-1][0]>0.) * 1.
        wx_min = bool(OF_lst[-1][0]<0.) * 1.

        wx_delta_plus = bool(OF_lst[-1][0]>OF_lst[-2][0]) * 1.
        wx_delta_min = bool(OF_lst[-1][0]<OF_lst[-2][0]) * 1.
####        
        ref_wx_node_plus = bool(np.random.uniform()>=(1-prob_ref_wx[0]))


        ref_wy_node_min = bool(np.random.uniform()>=(1-prob_ref_wy[1]))
        wy_plus = bool(OF_lst[-1][1]>0.) * 1.
        wy_min = bool(OF_lst[-1][1]<0.) * 1.

        wy_delta_plus = bool(OF_lst[-1][1]>OF_lst[-2][1]) * 1.
        wy_delta_min = bool(OF_lst[-1][1]<OF_lst[-2][1]) * 1.
####        
        ref_wy_node_plus = bool(np.random.uniform()>=(1-prob_ref_wy[0]))
####

        x = torch.tensor([ref_div_node_min, D_plus, D_min, D_delta_plus, D_delta_min, ref_div_node_plus, ref_wx_node_min, wx_plus, wx_min, wx_delta_plus, wx_delta_min, ref_wx_node_plus, ref_wy_node_min, wy_plus, wy_min, wy_delta_plus, wy_delta_min, ref_wy_node_plus])

        return x
    def spike_decoder(self, spike_array):

        thrust = spike_array[0] - spike_array[1]
        pitch = spike_array[2] - spike_array[3]
        roll = spike_array[4] - spike_array[5]

        return (thrust, pitch,  roll)

    def objective_function_single(self, model, ref_x, ref_y, ref_z):  # add prob here
        fig = plt.figure(figsize=(8,8))
        ax = fig.gca(projection='3d')
        steps=100000
        
        mav_model = model
        # self.environment.ref_div = prob_ref*2
        self.environment.ref_x = ref_x
        self.environment.ref_y = ref_y
        self.environment.ref_z = ref_z

        self.environment.reset()
        act_x = []
        act_y = []
        act_z = []

        self.environment.calc_time_to_point()

        reward_cum = 0
        for step in range(steps):
            self.environment.calc_prob_encoding_z()
            self.environment.calc_prob_encoding_x()
            self.environment.calc_prob_encoding_y()

            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], self.environment.encoding_z, self.environment.encoding_x, self.environment.encoding_y)

            array = mav_model(encoded_input.float()) 
            control_input = self.spike_decoder(array.detach().numpy())
            print(control_input) #control_input[1]
            divs, done, _ = self.environment.step(np.asarray([control_input[2], control_input[1], control_input[0]]))
            # divs, reward, done, _, _ = self.environment.step(np.asarray([0., 0., 0.]))
            # self.environment._get_reward()    
            # self.environment.render()
            if done:
                break
            
            act_x.append(self.environment.state[0][0])
            act_y.append(self.environment.state[1][0])
            act_z.append(self.environment.state[2][0])
            # divs_lst.append(self.environment.thrust_tc)
        
        # plt.plot(ref_horizontal,ref_vertical, c='#93a6f5')
        # plt.plot(ref_horizontal_uncoupled,ref_vertical, c='#e89725')
        ax.plot(act_x,act_y,act_z, c='#93a6f5', alpha=0.5)
        
        plt.ylabel('height (m)')
        plt.xlabel('distance (m)')
        plt.title('Divergence: ')
        # plt.savefig('25-08meeting.png')
        mav_model.reset()    
        reward_cum = self.environment.reward
        print(reward_cum)
        return reward_cum

    def objective_function_multiple(self, model, ref_x, ref_y, ref_z, runs):  # add prob here
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
            self.environment.ref_x = ref_x
            self.environment.ref_y = ref_y
            self.environment.ref_z = ref_z

            self.environment.reset()
            ref_vertical = []
            ref_horizontal = []
            act_vertical = []
            act_horizontal = []
            ref_horizontal_uncoupled = []
            reward_cum = 0
            for step in range(steps):
                encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], self.environment.encoding_z, self.environment.encoding_x, self.environment.encoding_y)

                array = mav_model(encoded_input.float()) 
                control_input = self.spike_decoder(array.detach().numpy())
                divs, done, _ = self.environment.step(np.asarray([control_input[2], control_input[1], control_input[0]]))
                # divs, reward, done, _, _ = self.environment.step(np.asarray([0., 0., 0.]))
                # self.environment._get_reward()    
                # self.environment.render()
                if done:
                    break
                ref_horizontal.append(self.environment.forward_reward)
                ref_vertical.append(self.environment.height_reward)
                act_horizontal.append(self.environment.state[0][0])
                act_vertical.append(self.environment.state[2][0])


                # print(self.environment.state[3][0],control_input[1])
                # ref_horizontal_uncoupled.append(self.environment.forward_reward_adm)
                
                # plt.plot(ref_horizontal_uncoupled,ref_vertical, c='#e89725')
                # divs_lst.append(self.environment.thrust_tc)
            if act_horizontal[-1]<nearest_traj:
                nearest_traj = act_horizontal[-1]
                nearest_traj_int = len(all_runs_vertical)

            if act_horizontal[-1]>furthest_traj:
                furthest_traj = act_horizontal[-1]
                furthest_traj_int = len(all_runs_vertical)
                
            all_runs_vertical.append(act_vertical)
            all_runs_horizontal.append(act_horizontal)

            ax.plot(np.arange(len(act_horizontal))*0.02,act_horizontal,act_vertical, c='#93a6f5', alpha=0.1)
            # print(act_horizontal)

        pad_vertical = len(max(all_runs_vertical, key=len))
        all_runs_matrix_vertical = np.array([i + [0]*(pad_vertical-len(i)) for i in all_runs_vertical])
        # print(all_runs_matrix_vertical)
        min_vert = np.amin(all_runs_matrix_vertical, axis=0)
        min_vert = min_vert[min_vert!=0]
        max_vert = np.amax(all_runs_matrix_vertical, axis=0)
        max_vert = max_vert[max_vert!=0]
        
        pad_horizontal = len(max(all_runs_horizontal, key=len))
        all_runs_matrix_horizontal = np.array([i + [0]*(pad_horizontal-len(i)) for i in all_runs_horizontal])
        min_hor = np.amin(all_runs_matrix_horizontal, axis=0)
        min_hor = min_hor[min_hor!=0]
        max_hor = np.amax(all_runs_matrix_horizontal, axis=0)
        max_hor = max_hor[max_hor!=0]

        # print(min_vert)
        # min_length_plots_min = min([len(min_vert),len(min_hor)])
        # plt.plot(min_hor[:min_length_plots_min],min_vert[:min_length_plots_min], c='#2b54ff')
        # min_length_plots_max = min([len(max_vert),len(max_hor)])
        # plt.plot(max_hor[:min_length_plots_max],max_vert[:min_length_plots_max],  c='#2b54ff')
        # print(min_length_plots_max, min_length_plots_min)


        # plt.plot(all_runs_horizontal[nearest_traj_int],all_runs_vertical[nearest_traj_int], c='#2b54ff')
        ax.plot(np.arange(len(all_runs_horizontal[nearest_traj_int]))*0.02,all_runs_horizontal[nearest_traj_int],all_runs_vertical[nearest_traj_int], c='#2b54ff', alpha=0.9)
        # plt.plot(all_runs_horizontal[furthest_traj_int],all_runs_vertical[furthest_traj_int],  c='#2b54ff')
        ax.plot(np.arange(len(all_runs_horizontal[furthest_traj_int]))*0.02,all_runs_horizontal[furthest_traj_int],all_runs_vertical[furthest_traj_int],  c='#2b54ff', alpha=0.9)

        # plt.plot(ref_horizontal_uncoupled,ref_vertical, c='#e89725')
        # plt.plot(ref_horizontal,ref_vertical, c='#eb9e34')
        ax.plot(np.arange(len(ref_horizontal))*0.02,ref_horizontal,ref_vertical,  c='#ffa72b', alpha=0.9)
        print(ref_horizontal[-1], ref_vertical[-1])
        print(act_horizontal[-1], act_vertical[-1])
        ax.set_zlabel('height (m)')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('distance (m)')
        plt.title('Divergence: ')
        ax.view_init(15, 90)
        plt.savefig('02-09meeting.png')
        print(self.environment.state[3],self.environment.state[5])
        mav_model.reset()    
        reward_cum = self.environment.reward
        return reward_cum



with open('testing_decreasedCMAES_3D.pkl', 'rb') as f:
    neat_class = dill.load(f)

# neat_class.species[neat_class.best_species].genomes[neat_class.best_genome]
neuron_matrix = find_all_routes(neat_class.best_genome)
neuron_matrix = clean_array(neuron_matrix)
model = place_weights(neuron_matrix, neat_class.best_genome)
network_viz = draw_net(neat_class.best_genome)
environment = Quadhover()
objective = objective(model, environment=environment)
# objective.objective_function_multiple(model, 0.8, 0.8, 1)
score = objective.objective_function_single(model, 1., 1., 1.)
print(model.state_dict())
print(neat_class.best_genome.fitness, neat_class.best_genome)


network_viz.view()
# print(neat_class.species[neat_class.best_species].genomes[neat_class.best_genome].fitness, neat_class.best_genome)



# draw_nn = DrawNN(model)
# draw_nn.draw()

# species = 0
# genome = 0

# network_viz = draw_net(neat_class.species[species].genomes[genome])
# network_viz.view()


# neuron_matrix = find_all_routes(neat_class.species[species].genomes[genome])
# neuron_matrix = clean_array(neuron_matrix)
# model = place_weights(neuron_matrix, neat_class.species[species].genomes[genome])
# network_viz = draw_net(neat_class.species[species].genomes[genome])
# environment = Quadhover()
# objective = objective(model, environment=environment)
# score = objective.objective_function_single(model, 1., 1., 1.)
# network_viz.view()
# print(neat_class.species[species].genomes[genome].fitness, neat_class.best_genome)
# print(score)

# draw_nn = DrawNN(model)
# draw_nn.draw()

# wanneer een tweede lange keten met geen enkele neuron onderdeel is van de eerste, wat dan?
# de eerste in de serie kan niet geplaatst worden
