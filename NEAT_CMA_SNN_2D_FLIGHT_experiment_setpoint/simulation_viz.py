from cProfile import label
from network_viz import draw_net
from species import find_all_routes, clean_array, place_weights
from env_3d_var_D_setpoint import LandingEnv3D as Quadhover
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
        fig = plt.gcf()
        fig.set_size_inches(28.5, 10.5)
        for i in range(spikes.shape[0]):
        #     print(len(spikes[0].reshape(-1)))
        #     print(len([(1./spikes.shape[0])*i]*spikes.shape[1]))
            # print(np.linspace(0.0, t, spikes.shape[1]), np.linspace(0.0, t, spikes.shape[1]).shape, len(spikes[0].reshape(-1)))
            plt.scatter(np.linspace(0.0, t, spikes.shape[1]), [(1.)*(spikes.shape[0]-i)]*spikes.shape[1], s=spikes[i,:].reshape(-1))
        plt.xlabel('time (s)')
        plt.ylabel('Encoding node')
        
        fig.savefig('test2png.png', dpi=100)         

        # plt.savefig('spike_analysis.png')
        

    def spike_encoder_div(self, OF_lst, prob_ref_div, prob_ref_wx):
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
####

        x = torch.tensor([ref_div_node_plus, D_plus, D_min, D_delta_plus, D_delta_min, ref_div_node_minus, ref_wx_node_plus, wx_plus, wx_min, wx_delta_plus, wx_delta_min, ref_wx_node_minus])

        return x
    def spike_decoder(self, spike_array):


        thrust = spike_array[0] - spike_array[1]
        pitch = spike_array[2] - spike_array[3]

        return (thrust, pitch)

    def objective_function_single(self, model, ref_div, ref_wx):  # add prob here

        steps=100000
            
        mav_model = model
        self.environment.ref_div = ref_div
        self.environment.ref_wx = ref_wx
        self.environment.reset()

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
            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_z, prob_x)

            array = mav_model(encoded_input.float()) 
            control_input = self.spike_decoder(array.detach().numpy())
            divs, reward, done, _, _ = self.environment.step(np.asarray([0., control_input[1], control_input[0]]))
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

    def objective_function_multiple(self, model, ref_div, ref_wx, runs):  # add prob here
        all_runs_vertical = []
        all_runs_horizontal = []

        nearest_traj = 100.
        nearest_traj_int = 0
        furthest_traj = 0.
        furthest_traj_int = 0
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca(projection='3d')

        all_touchdowns_z = []
        all_touchdowns_x = [] 
        for run in range(runs):

            steps=100000
            
            mav_model = model
            self.environment.ref_div = ref_div
            self.environment.ref_wx = ref_wx
            self.environment.reset()
            ref_vertical = []
            ref_horizontal = []
            act_vertical = []
            act_horizontal = []

            spikes = np.zeros((12,1))

            output_thrust = []
            output_thrust_actual = []

            input_control = []
            reward_cum = 0
            for step in range(steps):
                prob_x = self.environment.x_prob()
                prob_z = self.environment.z_prob()
                encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_z, prob_x)

                array = mav_model(encoded_input.float()) 
                # print(encoded_input.float().numpy().shape, encoded_input.float().numpy().reshape(12,1))
                spikes = np.hstack((spikes, encoded_input.float().numpy().reshape(12,1)))
                control_input = self.spike_decoder(array.detach().numpy())
                output_thrust.append(control_input[0])
                output_thrust_actual.append(self.environment.state[-1][0])
                divs, reward, done, _, _ = self.environment.step(np.asarray([0., control_input[1], control_input[0]]))
                # divs, reward, done, _, _ = self.environment.step(np.asarray([0., 0., 0.]))
                # self.environment._get_reward()    
                # self.environment.render()
                if done:
                    break
                ref_horizontal.append(self.environment.forward_reward)
                ref_vertical.append(self.environment.height_reward)
                act_horizontal.append(self.environment.state[0][0])
                act_vertical.append(self.environment.state[2][0])
                # input_control.append(control_input[0])
                # print(self.environment.state[2][0])


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

            ax.plot(np.arange(len(act_horizontal))*0.005,act_horizontal,act_vertical, c='#93a6f5', alpha=0.5)
            # print(act_horizontal)

            all_touchdowns_z.append(self.environment.state[5][0])
            all_touchdowns_x.append(self.environment.state[3][0])

        # plt.plot(all_runs_horizontal[nearest_traj_int],all_runs_vertical[nearest_traj_int], c='#2b54ff')
        
        # plt.plot(all_runs_horizontal[furthest_traj_int],all_runs_vertical[furthest_traj_int],  c='#2b54ff')
        # ax.plot(np.arange(len(all_runs_horizontal[furthest_traj_int]))*0.02,all_runs_horizontal[furthest_traj_int],all_runs_vertical[furthest_traj_int],  c='#2b54ff', alpha=0.9)

        # plt.plot(ref_horizontal_uncoupled,ref_vertical, c='#e89725')
        # plt.plot(ref_horizontal,ref_vertical, c='#eb9e34')

        # ax.plot(np.arange(len(all_runs_horizontal[nearest_traj_int]))*0.01,all_runs_horizontal[nearest_traj_int],all_runs_vertical[nearest_traj_int], c='#2b54ff', alpha=0.9, label='Actual trajectory')
        # ax.plot(np.arange(len(ref_horizontal))*0.01,ref_horizontal,ref_vertical,  c='#ffa72b', alpha=0.9, label='Reference trajectory')
        # print(ref_horizontal[-1], ref_vertical[-1])
        # print(act_horizontal[-1], act_vertical[-1])
        ax.set_zlabel('height (m)')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('distance (m)')
        plt.title('Divergence: '+ str(ref_div))
        ax.view_init(5, 90)
        ax.legend()
        plt.savefig('11-10meeting.png')
        plt.show()

        # plt.scatter(all_touchdowns_x, all_touchdowns_z)
        # plt.xlabel('Vx (m/s)')
        # plt.ylabel('Vz (m/s)')
        # plt.savefig('touchdownanalysis.png')
        # plt.close()
        # plt.plot(input_control)
        print(self.environment.state[3],self.environment.state[5])
        mav_model.reset()    
        reward_cum = self.environment.reward
        print('reward:', reward_cum, self.environment.t)
        # self.visualize_spikes(spikes,  self.environment.t)


        plt.plot(np.asanyarray(output_thrust), label='Thrust input')
        plt.plot(np.asanyarray(output_thrust_actual)-9.81, label='Thrust actual')
        plt.legend()
        plt.title('Mav thrust control input')
        plt.xlabel('timestep (0.005 s)')
        plt.ylabel('thrust (g)')
        plt.savefig('thrustmav.png')
        return reward_cum
# testing_decreasedCMAES_2D_fast_test for 2D hard landing
with open('testing_decreasedCMAES_2D_fast_test_nocma.pkl', 'rb') as f:
    neat_class = dill.load(f)

# neat_class.species[neat_class.best_species].genomes[neat_class.best_genome]
neuron_matrix = find_all_routes(neat_class.best_genome)
neuron_matrix = clean_array(neuron_matrix) 
model = place_weights(neuron_matrix, neat_class.best_genome)
network_viz = draw_net(neat_class.best_genome)
environment = Quadhover()
objective = objective(model, environment=environment)
objective.objective_function_multiple(model, 0.2, 0.2, 1)
# objective.objective_function_single(model, 0.5, 0.5)
print(model.state_dict())
print(neat_class.best_genome.fitness, neat_class.best_genome)




network_viz.view()
# print(neat_class.species[neat_class.best_species].genomes[neat_class.best_genome].fitness, neat_class.best_genome)



# draw_nn = DrawNN(model)
# draw_nn.draw()

# species = 0
# genome = 1


# neuron_matrix = find_all_routes(neat_class.species[species].genomes[genome])
# neuron_matrix = clean_array(neuron_matrix)
# model = place_weights(neuron_matrix, neat_class.species[species].genomes[genome])
# network_viz = draw_net(neat_class.species[species].genomes[genome])
# environment = Quadhover()
# objective = objective(model, environment=environment)
# objective.objective_function_multiple(model, 0.5, 0.5, 10)
# network_viz.view()
# print(neat_class.species[species].genomes[genome].fitness, neat_class.best_genome)

# draw_nn = DrawNN(model)
# draw_nn.draw()
