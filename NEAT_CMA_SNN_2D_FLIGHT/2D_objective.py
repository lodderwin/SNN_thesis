import torch
import numpy as np
import matplotlib.pyplot as plt
from env_3d_var import LandingEnv3D

class objective:
    def __init__(self, model, environment):
        # self.model = model
        # self.model = self.model.to(device=device)

        self.environment = environment

        self.thrust = 0

#     def spike_encoder_div(self, OF_lst, prob_ref_div, prob_ref_wx):
#         #current encoder
#         D_plus = bool(OF_lst[-1][-1]>0.) * OF_lst[-1][-1]
#         D_min = bool(OF_lst[-1][-1]<0.) * OF_lst[-1][-1]

#         D_delta_plus = bool(OF_lst[-1][-1]>OF_lst[-2][-1]) * (OF_lst[-1][-1] - OF_lst[-2][-1])
#         D_delta_min = bool(OF_lst[-1][-1]<OF_lst[-2][-1]) * (OF_lst[-1][-1] - OF_lst[-2][-1])
# ####        
#         ref_div_node = bool(np.random.uniform()>=(1-prob_ref_div))
# ####

#         wx_plus = bool(OF_lst[-1][0]>0.) * OF_lst[-1][0]
#         wx_min = bool(OF_lst[-1][0]<0.) * OF_lst[-1][0]

#         wx_delta_plus = bool(OF_lst[-1][0]>OF_lst[-2][0]) * (OF_lst[-1][0] - OF_lst[-2][0])
#         wx_delta_min = bool(OF_lst[-1][0]<OF_lst[-2][0]) * (OF_lst[-1][0] - OF_lst[-2][0])
# ####        
#         ref_wx_node = bool(np.random.uniform()>=(1-prob_ref_wx))
# ####

#         x = torch.tensor([D_plus, D_min, D_delta_plus, D_delta_min, ref_div_node, wx_plus, wx_min, wx_delta_plus, wx_delta_min, ref_wx_node])

#         return x

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

    # def spike_decoder(self, spike_array):

        # self.thrust = spike_array[0] + spike_array[1]



    def objective_function(self, ref_div, ref_wx):  # add prob here
        steps=100000
        
        # prob_ref = np.random.uniform(0.5, 1.5)*0.5
        # mav_model = build_model(x, self.model)

        # mav_model = self.model
    
        # self.environment.ref_div = prob_ref*2
        self.environment.ref_div = ref_div
        self.environment.ref_wx = ref_wx
        prob_ref_div = self.environment.ref_div/2.
        prob_ref_wx = self.environment.ref_wx/2.

        self.environment.reset()
        divs_lst = []
        ref_div_lst = []
        # print(mav_model.state_dict())

        reward_cum = 0
        for step in range(steps):
            encoded_input = self.spike_encoder_div(list(self.environment.obs)[-2:], prob_ref_div, prob_ref_wx)
            
        
            print(encoded_input) 
            # thrust_setpoint = mav_model(encoded_input.float())        
            # divs, reward, done, _, _ = self.environment.step(thrust_setpoint.detach().numpy())
            divs, reward, done, _, _ = self.environment.step(np.asarray([0., 0.01, 0.01]))
            to_append = self.environment.state[2]
            divs_lst.append(to_append[0])  #this zero is holy
            self.environment._get_reward()    
            if done:
                break
            
            ref_div_lst.append(self.environment.height_reward)
            # divs_lst.append(self.environment.thrust_tc)
        
        # mav_model.reset()    

        # # time.sleep(0.1)
        plt.plot(divs_lst, c='#4287f5')
        # plt.plot(ref_lst, c='#f29b29')

        
        # plt.ylabel('height (m)')
        # plt.xlabel('timesteps (0.02s)')
        # plt.title('Divergence: '+ str(ref_div))
        # # plt.plot(divs_lst)
        
        # # figure(figsize=(8, 6), dpi=80)
        # # plt.show()
        # plt.savefig('pres_1.png')
        # # plt.show()
        reward_cum = self.environment.reward
        print(divs_lst)
        return reward_cum

objective = objective('asdf', LandingEnv3D())
reward = objective.objective_function(1,1)