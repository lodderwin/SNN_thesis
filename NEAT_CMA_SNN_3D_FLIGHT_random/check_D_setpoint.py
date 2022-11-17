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

    def visualize_spikes(self, spikes, t):
        for i in range(spikes.shape[0]):
        #     print(len(spikes[0].reshape(-1)))
        #     print(len([(1./spikes.shape[0])*i]*spikes.shape[1]))
            # print(np.linspace(0.0, t, spikes.shape[1]), np.linspace(0.0, t, spikes.shape[1]).shape, len(spikes[0].reshape(-1)))
            plt.scatter(np.linspace(0.0, t, spikes.shape[1]), [(1.)*(spikes.shape[0]-i)]*spikes.shape[1], s=spikes[i,:].reshape(-1))

        plt.xlabel('time (s)')
        plt.ylabel('Encoding node')
        plt.savefig('spike_analysis.png')

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

        x = torch.tensor([ref_wx_node_min, wx_plus, wx_min, wx_delta_plus, wx_delta_min, ref_wx_node_plus, ref_div_node_min, D_plus, D_min, D_delta_plus, D_delta_min, ref_div_node_plus, ref_wy_node_min, wy_plus, wy_min, wy_delta_plus, wy_delta_min, ref_wy_node_plus])

        return x
    def spike_decoder(self, spike_array):


        pitch = spike_array[0] - spike_array[1]
        thrust = spike_array[2] - spike_array[3]
        roll = spike_array[4] - spike_array[5]

        return (pitch, thrust,  roll)

    def objective_function_single(self, model, ref_div, ref_wx, ref_wy):  # add prob here

        steps=100000
        
        mav_model = model
        # self.environment.ref_div = prob_ref*2
        self.environment.ref_z = ref_div
        self.environment.ref_wx = ref_wx
        self.environment.ref_wy = ref_wy
      
        self.environment.coordinates = [ref_wx, ref_wy, ref_div]


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
            divs, done, _ = self.environment.step(np.asarray([control_input[2], control_input[0], control_input[1]]))
            # divs, reward, done, _, _ = self.environment.step(np.asarray([0., 0., 0.]))
            # self.environment._get_reward()    
            # self.environment.render()
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

    def objective_function_multiple(self, model, ref_z, ref_y, ref_x, runs):  # add prob here

        fig = plt.figure(figsize=(5,5))
        ax = fig.gca(projection='3d')

        
        # xy hovering  ++ +- -- -+ a
        # z is randomized between 1 and 5.

        
        for run in range(runs):

            steps=100000
            
            mav_model = model
            # self.environment.ref_div = prob_ref*2
            self.environment.ref_x = ref_x
            self.environment.ref_y = ref_y
            self.environment.ref_z = ref_z
            self.environment.coordinates = [ref_x, ref_y, ref_z]

            self.environment.reset()
            ref_vertical = []
            ref_horizontal = []
            act_vertical = []
            act_horizontal = []
            act_side = []

            d_x = []

            spikes = np.zeros((18,1))

            self.environment.calc_start()

            input_control = []
            reward_cum = 0
            for step in range(steps):
                self.environment.calc_D_const()
                print(self.environment.D_setpoint_x, self.environment.D_setpoint_y, self.environment.D_setpoint_z)
                self.environment.encoding_x = self.environment.x_prob()
                self.environment.encoding_z = self.environment.z_prob()
                self.environment.encoding_y = self.environment.y_prob()
                encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], self.environment.encoding_z, self.environment.encoding_x, self.environment.encoding_y)
                d_x.append(self.environment.D_setpoint_x)
                array = mav_model(encoded_input.float()) 
                control_input = self.spike_decoder(array.detach().numpy())
                spikes = np.hstack((spikes, encoded_input.float().numpy().reshape(18,1)))
                divs, done, _ = self.environment.step(np.asarray([control_input[2], control_input[0], control_input[1]]))
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
        
            ax.plot(act_horizontal,act_side,act_vertical, c='#93a6f5', alpha=0.9)
        
        # print(ref_horizontal[-1], ref_vertical[-1])
        # print(act_horizontal[-1], act_vertical[-1])
        ax.set_zlabel('height (m)')
        ax.set_xlabel('distance x (m)')
        ax.set_ylabel('distance y (m)')
        plt.title('Divergence: ')
        ax.view_init(15, 45)  #(15,90)
        ax.legend()

        plt.show()
        # plt.savefig('2709meeting.png')
        # plt.plot(input_control)
        print(self.environment.state[3],self.environment.state[5])
        mav_model.reset()    
        reward_cum = self.environment.reward
        print('reward:', reward_cum, self.environment.t)
        # self.visualize_spikes(spikes, self.environment.t)
        plt.plot(d_x)
        return reward_cum

with open('testing_decreasedCMAES_3D_new_control_sys.pkl', 'rb') as f:
    neat_class = dill.load(f)

# neat_class.species[neat_class.best_species].genomes[neat_class.best_genome]
neuron_matrix = find_all_routes(neat_class.best_genome)
neuron_matrix = clean_array(neuron_matrix) 
model = place_weights(neuron_matrix, neat_class.best_genome)
network_viz = draw_net(neat_class.best_genome)
environment = Quadhover()
objective = objective(model, environment=environment)
# objective.objective_function_multiple(model, 1)
objective.objective_function_multiple(model, 2.5, 1., 1., 1)
print(model.state_dict())
print(neat_class.best_genome.fitness, neat_class.best_genome)


network_viz.view()
# print(neat_class.species[neat_class.best_species].genomes[neat_class.best_genome].fitness, neat_class.best_genome)



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