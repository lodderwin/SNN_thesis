from cProfile import label
from collections import deque
import matplotlib.pyplot as plt

import numpy as np
import copy


class LandingEnv3D:
    """
    3D MAV control environment based on the formulation in Li et al, 2020.

    'Visual model-predictive localization for computationally efficient autonomous racing of a 72-g drone'
    Source: http://pure.tudelft.nl/ws/portalfiles/portal/73036228/rob.21956.pdf

    Yaw control (psi) has been taken out!
    Body frame: right-handed, Z upwards, X aligned with drone body (not with any arm)
    World frame: right-handed, Z upwards
    """

    # Constants
    G = 9.81

    def __init__(
        self,
        ref_x=0.0,
        ref_y=0.0,
        ref_z=2.5,
        obs_setpoint=[0, 0, 0.5],
        obs_weight=[1.0] * 3,
        #changed from 0 to 1
        obs_delay=1,
        obs_bias=[0.0] * 3,
        obs_noise_std=0.1,
        obs_noise_p_std=0.1,
        wind_std=0.0,
        state_0=[0, 0, 2.5, 0, 0, 0, 0, 0, 0, G],
        h_blind=6.,
        gains=[6, 6, 3.0],
        dt=0.005,
        seed=None,
        # act_high=[1 * np.pi] * 2 + [0.4 * G],  #why?
        act_high=[1 * 0.785] * 2 + [0.4 * G],  #why?
        state_bounds=[[-5, -5, 0.1], [5, 5, 5]],
        time_bound=30,
        default_D=0.2,
    ):
        # Parameter dicts:
        # - public for randomizable parameters
        # - protected for non-randomizable ones
        self.param = {
            "obs setpoint": np.array(obs_setpoint),
            "obs weight": np.array(obs_weight),
            "obs delay": obs_delay,
            "obs bias": obs_bias,
            "obs noise std": obs_noise_std,
            "obs noise p std": obs_noise_p_std,
            "wind std": wind_std,
            "state 0": np.array(state_0).reshape(-1, 1),
            "gains": np.array(gains).reshape(-1, 1),
            "dt": dt,
            "seed": seed,
        }
        self._param = {
            "act high": np.array(act_high),
            "state bounds": np.array(state_bounds),
            "time bound": time_bound,
        }

        # Create local RNG
        self.rng = np.random.default_rng(seed)

        # Do checks on arguments
        self._checks()

        # Reset to get initial observation and allocate arrays
        self.reset()
        self.max_h = state_bounds[1][2]


        self.ref_x = ref_x
        self.ref_y = ref_y
        self.ref_z = ref_z
        self.start_coordinates = [0., 0., 2.5]
        self.coordinates = [self.ref_x, self.ref_y, self.ref_z]

        self.delta_x = self.coordinates[0] - self.start_coordinates[0]
        self.delta_y = self.coordinates[1] - self.start_coordinates[1]
        self.delta_z = self.coordinates[2] - self.start_coordinates[2]
        # build trajectory.
        # fly square with alternating height. if within 1 m, get new coordinates

        #TODO: create probability function
        self.update_rate_position = 0.005 #--> changes probability nodes
        # probability_nodes x distance, y probability

        self.encoding_x = [0.5, 0.5]
        self.encoding_y = [0.5, 0.5]
        self.encoding_z = [0.5, 0.5]

        self.D_setpoint_x = 0.0
        self.D_setpoint_y = 0.0
        self.D_setpoint_z = 0.0

        self.default_D = default_D
        # self.calc_time_to_point()
        # print(self.ref_x, self.ref_y, self.ref_z, self._param["time bound"])

        self.reward = 0
        self.height_reward = self.start_coordinates[2]
        self.forward_reward = self.start_coordinates[0]
        self.side_reward = self.start_coordinates[1]

        self.h_blind = h_blind

    def calc_start(self):
        self.height_reward = self.start_coordinates[2]
        self.forward_reward = self.start_coordinates[0]
        self.side_reward = self.start_coordinates[1]
        self.delta_x = self.coordinates[0] - self.start_coordinates[0]
        self.delta_y = self.coordinates[1] - self.start_coordinates[1]
        self.delta_z = self.coordinates[2] - self.start_coordinates[2]

    def x_prob(self):
        # print('Xh', self.state[3]/(2*self.state[2]))
        variable = self.D_setpoint_x - self.state[3][0]/(2*self.state[2][0]) 
        if variable>=0.:
            if variable>1.:
                variable = 1.
            prob = np.exp(-1./((variable)/0.1))/1. 
            return (prob, 0.)
        elif variable<0.:
            if variable<-1.:
                variable = -1.
            prob = np.exp(1./((1*variable)/0.1))/1. 
            return  (0., prob )

    def z_prob(self):
        # print('Xh', self.state[3]/(2*self.state[2]))
        variable = self.D_setpoint_z - self.state[5][0]/(2.*self.state[2][0]) 
        if variable>=0.:
            if variable>1.:
                variable = 1.
            prob = np.exp(-1./((variable)/0.1))/1. 
            return (prob, 0.)
        elif variable<0.:
            if variable<-1.:
                variable = -1.
            prob = np.exp(1./((1*variable)/0.1))/1. 
            return  (0., prob )

    def y_prob(self):
        variable = self.D_setpoint_y - self.state[4][0]/(2.*self.state[2][0]) 
        if variable>=0.:
            if variable>1.:
                variable = 1.
            prob = np.exp(-1./((variable)/0.1))/1. 
            return (prob, 0.)
        elif variable<0.:
            if variable<-1.:
                variable = -1.
            prob = np.exp(1./((1*variable)/0.1))/1. 
            return  (0., prob )


    def calc_D_const(self):
        if self.delta_x >=0.:
            self.D_setpoint_x = self.default_D
            if self.coordinates[0] - self.state[2][0]<self.state[0][0]<self.coordinates[0] + self.state[2][0]:
                self.D_setpoint_x = (self.coordinates[0] - self.state[0][0]) * ((2.*self.default_D)/(2.*self.state[2][0]))
            if self.coordinates[0] + self.state[2][0]<self.state[0][0]:
                self.D_setpoint_x = -self.default_D
        if self.delta_x < 0.:
            self.D_setpoint_x = -self.default_D
            if self.coordinates[0] - self.state[2][0]<self.state[0][0]<self.coordinates[0] + self.state[2][0]:
                self.D_setpoint_x = (self.coordinates[0] - self.state[0][0]) * ((2.*self.default_D)/(2.*self.state[2][0]))
            if self.coordinates[0] - self.state[2][0]>self.state[0][0]:
                self.D_setpoint_x = self.default_D
        

        if self.delta_y >=0.:
            self.D_setpoint_y = self.default_D
            if self.coordinates[1] - self.state[2][0]<self.state[1][0]<self.coordinates[1] + self.state[2][0]:
                self.D_setpoint_y = (self.coordinates[1] - self.state[1][0]) * ((2.*self.default_D)/(2.*self.state[2][0]))
            if self.coordinates[1] + self.state[2][0]<self.state[1][0]:
                self.D_setpoint_y = -self.default_D
        if self.delta_y < 0.:
            self.D_setpoint_y = -self.default_D
            if self.coordinates[1] - self.state[2][0]<self.state[1][0]<self.coordinates[1] + self.state[2][0]:
                self.D_setpoint_y = (self.coordinates[1] - self.state[1][0]) * ((2.*self.default_D)/(2.*self.state[2][0]))
            if self.coordinates[1] - self.state[2][0]>self.state[1][0]:
                self.D_setpoint_y = self.default_D

        if self.delta_z >=0.:
            self.D_setpoint_z = self.default_D
            if (self.coordinates[2] - 1.)<self.state[2][0]<(self.coordinates[2] + 1.):
                self.D_setpoint_z = (self.coordinates[2] - self.state[2][0]) * ((2.*self.default_D)/(2.*1))
            if self.coordinates[2] + 1<self.state[2][0]:
                self.D_setpoint_z = -self.default_D
        if self.delta_z < 0.:
            self.D_setpoint_z = -self.default_D
            if (self.coordinates[2] - 1.)<self.state[2][0]<(self.coordinates[2] + 1.):
                self.D_setpoint_z = (self.coordinates[2] - self.state[2][0]) * ((2.*self.default_D)/(2.*1))
            if self.coordinates[2] - 1>self.state[2][0]:
                self.D_setpoint_z = self.default_D

    def calc_time_to_point(self):
        self._param["time bound"] =  np.sqrt( np.sqrt(self.ref_x**2 + self.ref_y**2 + np.abs(self.ref_z-2.5)**2) )*4
        # print(self.param["time bound"])


    def _checks(self):
        assert self.param["obs setpoint"].shape[0] == 3
        assert self.param["obs delay"] >= 0 and isinstance(self.param["obs delay"], int)
        assert (
            self.param["obs noise std"] >= 0.0 and self.param["obs noise p std"] >= 0.0
        )
        assert self.param["wind std"] >= 0.0
        assert self.param["state 0"][2, 0] >= 0.0
        assert (self.param["gains"] >= 0.0).all()
        assert self.param["dt"] > 0.0
        assert (self._param["act high"] <= np.array([2 * np.pi] * 2 + [self.G])).all()
        assert (self._param["state bounds"][1] > self._param["state bounds"][0]).all()
        assert self._param["time bound"] > 0.0


    def set_param(self, param):
        # Check for keys
        assert set(self.param.keys()) >= set(param.keys())

        # Update and reseed (create new RNG again)
        self.param.update(param)
        self._checks()
        self.rng = np.random.default_rng(self.param["seed"])

    def step(self, act):
        # Check if already done last time
        # if self.done:
        #     return self.obs[self.steps], self.reward, self.done

        # Input act is in (-1, 1), scaled with 'act high'
        # Offset G later!
        self.act = (act.clip(-1.0, 1.0) * self._param["act high"]).reshape(-1, 1)



        # print(self.act)
        #TODO: add clamp for thrust and pitch

        # Action was taken based on previous observation/state, so now increment step
        self.steps += 1
        self.t += self.param["dt"]

        # Update state with forward Euler
        self.state += self._get_state_dot() * self.param["dt"]
        
        self.reward = self._get_reward() + self.reward
        self.done = self._check_out_of_bounds() | self._check_out_of_time() 

        # Clamp altitude to bounds (VERY important for reward because of 1/h)
        # TODO: make this nicer?
        if self.done:
            pos_min, pos_max = self._param["state bounds"]  
            self.state[0:3] = np.clip(
                self.state[0:3], pos_min.reshape(-1, 1), pos_max.reshape(-1, 1)
            )
            self.t = np.clip(self.t, 0.0, self._param["time bound"])
            # self.reward = self.reward + np.abs(self.state[5][0]*5)
            # self.reward = self.reward + np.abs(self.state[3][0]*5)
            speed_multiplier = 50
            # self.reward = self.reward + np.abs(self.state[4][0]*5)
            self.reward = self.reward + np.abs(self.ref_x - self.state[0][0]) + np.abs(self.ref_y - self.state[1][0]) + np.abs(self.ref_z - self.state[2][0])*2 +\
            np.abs(self.state[3][0]*speed_multiplier) + np.abs(self.state[4][0]*speed_multiplier)  + np.abs(self.state[5][0]*speed_multiplier) 
            # self.reward = self._get_reward()


        return self._get_obs(), self.done, self.height_reward

    def _get_state_dot(self):
        """
        State: [x, y, z, vx, vy, vz, phi, theta, psi, thrust]^T
        Actions: [phi^c, theta^c, psi^c = 0, thrust^c]^T
        """
        # Position
        # Add wind as velocity disturbance in world frame

        # don't pay attention to the dot label
        p_dot = self.state[3:6] + self._get_wind()
        # print(self.wind)
        # Velocity
        v_dot = (
            np.array([0, 0, -self.G]).reshape(-1, 1)
            + self._body2world() @ np.array([0, 0, self.state[9, 0]]).reshape(-1, 1)
            + self._body2world()
            @ np.diagflat([-0.5, -0.5, 0])
            @ self._world2body()
            @ self.state[3:6]
        )
        # Actions
        act = np.insert(self.act, 2, 0, axis=0)  # insert 0 psi^c  # for hovering, so change
        act[3, 0] += self.G  # offset G
        gains = np.insert(self.param["gains"], 2, 0, axis=0)  # and 0 gain
        act_dot = gains * (act - self.state[6:10])
        return np.vstack((p_dot, v_dot, act_dot))

    def _get_wind(self):
        """
        Wind that is correlated over time.
        """
        if self.param["wind std"] > 0.0:
            self.wind += (
                (
                    self.rng.normal(0.0, self.param["wind std"], size=self.wind.shape)
                    - self.wind
                )
                * self.param["dt"]
                / (self.param["dt"] + self.param["wind std"])
            )
        return self.wind

    def _get_obs(self):
        """
        Observation of ventral flows and divergence (visual observables).
        """
        # Compute observation
        self.obs_gt[0] = -self.state[3, 0] / np.maximum(1e-5, self.state[2, 0])  # /z !?
        # plus wx, wy, D due to rotations
        self.obs_gt[1] = -self.state[4, 0] / np.maximum(1e-5, self.state[2, 0])
        self.obs_gt[2] = -2 * self.state[5, 0] / np.maximum(1e-5, self.state[2, 0])

        wx_dot = (self.obs_gt[0] - self.wx_ph[0]) / self.param['dt']
        wy_dot = (self.obs_gt[1] - self.wy_ph[0]) / self.param['dt']
        div_dot = (self.obs_gt[2] - self.div_ph[0]) / self.param['dt']

        self.wx_ph[:] = [self.obs_gt[0], wx_dot]
        self.wy_ph[:] = [self.obs_gt[1], wy_dot]
        self.div_ph[:] = [self.obs_gt[2], div_dot]

        # Add bias + noise
        # TODO: can this be done vectorized?
        wx = (
            self.obs_gt[0]
            + self.rng.normal(self.param["obs bias"][0], self.param["obs noise std"])
            + abs(self.obs_gt[0]) * self.rng.normal(0.0, self.param["obs noise p std"])
        )
        wy = (
            self.obs_gt[1]
            + self.rng.normal(self.param["obs bias"][1], self.param["obs noise std"])
            + abs(self.obs_gt[1]) * self.rng.normal(0.0, self.param["obs noise p std"])
        )
        div = (
            self.obs_gt[2]
            + self.rng.normal(self.param["obs bias"][2], self.param["obs noise std"])
            + abs(self.obs_gt[2]) * self.rng.normal(0.0, self.param["obs noise p std"])
        )

        # Append to end of deque

        wx_dot = (wx - self.obs[-1][0]) / self.param['dt']
        wy_dot = (wy - self.obs[-1][1]) / self.param['dt']
        div_dot = (div - self.obs[-1][2]) / self.param['dt']
        
        self.obs.append(np.array([wx, wy, div, wx_dot, wy_dot, div_dot]))

        # return self.obs[0]

        # if self.state[2] > self.h_blind:
            # return np.zeros(6, dtype=np.float32)
        # else:
        return np.array(self.obs[0], dtype=np.float32)

    def _get_reward(self):
        # return (
        #     1
        #     - (
        #         np.abs(self.obs_gt - self.param["obs setpoint"])
        #         * self.param["obs weight"]
        #     ).sum()
        # ).clip(-1.0, 1.0)
        if self.t<1.0:
            # None
            return 0.
        else:
            z_delta = (self.D_setpoint_z*self.state[2][0]/1.)*self.param["dt"]
            x_delta = (self.D_setpoint_x*self.state[2][0]/1.)*self.param["dt"]
            y_delta = (self.D_setpoint_y*self.state[2][0]/1.)*self.param["dt"]

            height_reward = np.abs((self.state[2][0] - self.height_reward ) - z_delta)   #        *self.param["dt"]
            forward_reward = np.abs((self.state[0][0] - self.forward_reward) - x_delta) # 
            side_reward = np.abs((self.state[1][0] - self.side_reward) - y_delta) # 
            self.height_reward = copy.copy(self.state[2][0])  #something wrong with copy
            self.forward_reward = copy.copy(self.state[0][0])    #      *self.param["dt"]
            self.side_reward = copy.copy(self.state[1][0]) 
            return  forward_reward + height_reward + side_reward


        # speed_multiplier = 1
        # return np.abs(self.ref_x - self.state[0][0]) + np.abs(self.ref_y - self.state[1][0]) + np.abs(self.ref_z - self.state[2][0]) +\
        #     np.abs(self.state[3][0]*speed_multiplier) + np.abs(self.state[4][0]*speed_multiplier)  + np.abs(self.state[5][0]*speed_multiplier) 

    def reset(self, h0=5.0):
        # Initial state
        self.state = self.param["state 0"].copy() 
        # starting state: x, y, z, vx, vy, vz, phi, theta, psi, thrust
        self.wind = np.zeros((3, 1))  # wind in three directions
        self.act = np.zeros((3, 1))  #roll pitch thrust

        # Circular array for dealing with delayed observations
        self.obs = deque([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], maxlen=self.param["obs delay"] + 1)
        self.obs_gt = np.zeros(3)   #true obs with noise

        self.height_reward = h0
        self.forward_reward = 0.0
        self.x_reward = 0.
        self.reward = 0


        # Counting variables
        self.done = False
        self.steps = 0
        self.t = 0.0

        self.div_ph = np.array([0.0, 0.0], dtype=np.float32)
        self.wx_ph = np.array([0.0, 0.0], dtype=np.float32)
        self.wy_ph = np.array([0.0, 0.0], dtype=np.float32)

        # Fill up observations
        for _ in range(self.param["obs delay"] + 1):
            self._get_obs()

        return self.obs[0]

    def _body2world(self):
        return body2world(*self.state[6:9, 0])

    def _world2body(self):
        return world2body(*self.state[6:9, 0])

    def _clamp(self, value, minimum, maximum):
        return max(min(value, maximum), minimum)

    def _check_out_of_bounds(self):
        pos_min, pos_max = self._param["state bounds"]
        return (
            (self.state[0:3] <= pos_min.reshape(-1, 1))
            | (self.state[0:3] >= pos_max.reshape(-1, 1))
        ).any()

    def _check_out_of_time(self):
        return (self.steps + 1) * self.param["dt"] >= self._param["time bound"]

    def _check_projected_landing(self):
        return self.height_reward<0.01

    

    # def checkfunction(self, ref_div, ref_wx, ref_wy):

    #     D_x = []
    #     plt_traj = []
    #     self.ref_z = ref_div
    #     self.ref_wx = ref_wx
    #     self.ref_wy = ref_wy
      
    #     self.coordinates = [ref_wx, ref_wy, ref_div]
    #     self.calc_start()
    #     while not self.done:
    #         prob_y = self.y_prob()
    #         self.calc_D_const()
    #         self.done = self._check_out_of_bounds() | self._check_out_of_time() | self._check_projected_landing()
    #         self.t += self.param["dt"]


    #         # self.state[2][0] += -0.015
    #         # self.state[3][0] += 0.
    #         self.state[1][0] += 0.01

    #         # reward = self._get_reward()
    #         # self.reward = self.reward + reward
    #         # print('a', reward)
    #         # print(self.D_setpoint_z)
    #         D_x.append(prob_y)

    #         plt_traj.append(self.state[1][0])
            

    #     if self.done:
    #             pos_min, pos_max = self._param["state bounds"]  
    #             self.state[0:3] = np.clip(
    #                 self.state[0:3], pos_min.reshape(-1, 1), pos_max.reshape(-1, 1)
    #             )
    #             self.t = np.clip(self.t, 0.0, self._param["time bound"])
    #             self.reward = self.reward + np.abs(self.state[5][0]*10)
    #             self.reward = self.reward + np.abs(self.state[3][0]*10)
               
    #     x = np.arange(0, len(plt_traj), 1)
    #     plt.plot(plt_traj[:1000],D_x[:1000])
    #     # plt.fill_between(x, plt_traj, plt_traj_ref[:len(plt_traj)], color='#808080')
        
    #     # plt.ylabel('height (m)')
    #     # plt.xlabel('timesteps 0.02 (s)')
    #     # plt.legend()
    #     # plt.title('Constant divergence landing')
    #     # plt.savefig('show2D_ref.png')
    #     # plt.show()
        
    #     # print(self.reward)

def body2world(phi, theta, psi):
    """
    Rotation matrix from body frame to world frame.
    """
    c, s = np.cos, np.sin
    rot = np.array(
        [
            [
                c(psi) * c(theta),
                c(psi) * s(theta) * s(phi) - s(psi) * c(phi),
                c(psi) * s(theta) * c(phi) + s(psi) * s(phi),
            ],
            [
                s(psi) * c(theta),
                s(psi) * s(theta) * s(phi) + c(psi) * c(phi),
                s(psi) * s(theta) * c(phi) - c(psi) * s(phi),
            ],
            [-s(theta), c(theta) * s(phi), c(theta) * c(phi)],
        ]
    )
    return rot


def world2body(phi, theta, psi):
    """
    Rotation matrix from world frame to body frame.
    """
    return body2world(phi, theta, psi).T




    # check actual reward function by simulating straight line trajectories


# environment = LandingEnv3D()
# environment.reset()
# environment.checkfunction(3.5,-3.5,4.)

