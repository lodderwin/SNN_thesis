from env_3d_var import LandingEnv3D as Quadhover
from snn_pytorch_mod_double_output import SNN
import torch
import numpy as np
import dill
import matplotlib.pyplot as plt
import numpy as np
# from visualize import DrawNN

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

class objective:
    def __init__(self, model, environment):
        self.model = model
        self.environment = environment

        # mav_model = build_model_param(mav_model, 'weight')
        
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

    def objective_function_single(self, ref_div, ref_wx):  # add prob here

        steps=100000
        
        # print(mav_model.state_dict())
        # mav_model = build_model(x, self.model)

        mav_model = self.model
        self.environment.ref_div = ref_div
        self.environment.ref_wx = ref_wx
        prob_ref_div = self.environment.ref_div/2.
        prob_ref_wx = self.environment.ref_wx/2.


        

        self.environment.reset()
        ref_vertical = []
        ref_horizontal = []
        act_vertical = []
        act_horizontal = []
        ref_horizontal_uncoupled = []
        reward_cum = 0
        for step in range(steps):
            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_ref_div, prob_ref_wx)

            array = mav_model(encoded_input.float()) 
            control_input = self.spike_decoder(array.detach().numpy())
            print(control_input) #control_input[1]
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
            ref_horizontal_uncoupled.append(self.environment.forward_reward_adm)
            # divs_lst.append(self.environment.thrust_tc)
        
        plt.plot(ref_horizontal,ref_vertical, c='#93a6f5')
        plt.plot(act_horizontal,act_vertical,  c='#2b54ff')
        plt.plot(ref_horizontal_uncoupled,ref_vertical, c='#e89725')

        
        plt.ylabel('height (m)')
        plt.xlabel('distance (m)')
        plt.title('Divergence: '+ str(ref_div))
        # plt.savefig('25-08meeting.png')
        mav_model.reset()    
        reward_cum = self.environment.reward
        print(reward_cum, self.environment.state[0][0], self.environment.state[2][0] )
        return reward_cum

    def objective_function_multiple(self, ref_div, ref_wx, runs):  # add prob here
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
            
            mav_model = self.model
            # self.environment.ref_div = prob_ref*2
            self.environment.ref_div = ref_div
            self.environment.ref_wx = ref_wx
            prob_ref_div = self.environment.ref_div/2.
            prob_ref_wx = self.environment.ref_wx/2.

            self.environment.reset()
            ref_vertical = []
            ref_horizontal = []
            act_vertical = []
            act_horizontal = []
            ref_horizontal_uncoupled = []
            reward_cum = 0
            for step in range(steps):
                encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_ref_div, prob_ref_wx)

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
        plt.title('Divergence: '+ str(ref_div))
        ax.view_init(15, 90)
        # plt.savefig('02-09meeting.png')
        mav_model.reset()    
        reward_cum = self.environment.reward
        return reward_cum

with open('CMA_ES_0709.pkl', 'rb') as f:
    organize_training = dill.load(f)

environment = Quadhover()

model = organize_training.objective.model

x = organize_training.xmean
x_decay, x_thresh, x_weights = x[:organize_training.objective.N_decay], x[organize_training.objective.N_decay-1:organize_training.objective.N_threshold], x[organize_training.objective.N_threshold-1:]
model = build_model_param(x_decay, model, 'v_decay')
model = build_model_param(x_thresh, model, 'threshold')
model = build_model_param(x_weights, model, 'weight')

# model = build_model_param(organize_training.xmean, organize_training.objective.model, 'weight')



objective = objective(model, environment)
objective.objective_function_multiple(1., 1., 50)
# objective.objective_function_single(1., 1.)
# print(model.state_dict())


# print(neat_class.species[neat_class.best_species].genomes[neat_class.best_genome].fitness, neat_class.best_genome)



# draw_nn = DrawNN(model)
# draw_nn.draw()

# species = 0
# genome = 0


# neuron_matrix = find_all_routes(neat_class.species[species].genomes[genome])
# neuron_matrix = clean_array(neuron_matrix)
# model = place_weights(neuron_matrix, neat_class.species[species].genomes[genome])
# network_viz = draw_net(neat_class.species[species].genomes[genome])
# environment = Quadhover()
# objective = objective(model, environment=environment)
# objective.objective_function_multiple(model, 1., 1., 50)
# network_viz.view()
# print(neat_class.species[species].genomes[genome].fitness, neat_class.best_genome)

# draw_nn = DrawNN(model)
# draw_nn.draw()
